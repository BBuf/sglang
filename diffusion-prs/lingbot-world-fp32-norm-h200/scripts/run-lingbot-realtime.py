"""Native WebSocket multi-chunk evidence using SGLang's raw-frame CI collector."""
import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--label', required=True)
p.add_argument('--chunks', type=int, default=10)
p.add_argument('--port', type=int, default=23880)
p.add_argument('--master-port', type=int, default=23980)
args = p.parse_args()
root = Path('/campaign')
repo = Path(args.repo).resolve()
lock = (root/'gpu.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
os.environ.update(json.loads((root/'artifacts/lingbot-world-v2-cache-env.json').read_text()))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', PYTHONPATH=str(repo/'python'),
    SGLANG_DIFFUSION_SYNC_STAGE_PROFILING='1')
sys.path.insert(0, str(repo/'python'))
os.chdir(repo)
import sglang
assert Path(sglang.__file__).resolve().is_relative_to(repo)
subprocess.run(['git','diff','--quiet'], check=True)
subprocess.run(['git','diff','--cached','--quiet'], check=True)
selected = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
for _ in range(30):
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if not any(line.startswith(selected) for line in processes.splitlines()):break
    time.sleep(1)
else:raise RuntimeError('Assigned GPU remains busy; no process killed')
for port in (args.port,args.master_port):
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
out = root/'artifacts/lingbot-world-v2'/args.label
out.mkdir(parents=True,exist_ok=False)
(out/'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True)+str(sglang.__file__)+'\n')
model = 'robbyant/lingbot-world-v2-14b-causal-fast-diffusers'
command = ['sglang','serve','--model-path',model,'--num-gpus=1','--tp-size=1',
    '--ulysses-degree=1','--performance-mode=manual','--dit-layerwise-offload=false',
    '--dit-cpu-offload=false','--text-encoder-cpu-offload=true','--vae-cpu-offload=false',
    '--enable-torch-compile=false','--enable-breakable-cuda-graph=false',
    '--warmup-resolutions=832x480','--warmup-num-frames=9',
    '--host=127.0.0.1',f'--port={args.port}',f'--master-port={args.master_port}']
(out/'command.json').write_text(json.dumps(command,indent=2))
from sglang.multimodal_gen.test.server.realtime_consistency import (
    collect_realtime_output, encode_realtime_frames_to_mp4,
)
from PIL import Image

payload = dict(type='init',model=model,
    prompt='A slow aerial orbit around a pastel island hotel in the ocean.',
    first_frame=(root/'artifacts/input-media/longlive2-cat.png').read_bytes(),
    size='832x480',fps=16,num_frames=9,seed=42,num_inference_steps=4,
    guidance_scale=1,quality='lossless',max_chunks=args.chunks,
    realtime_output_format='raw',realtime_output_pacing=False,
    condition_inputs={'camera_actions':[['w'] for _ in range(9*args.chunks+32)]})
saved_payload = payload | {'first_frame':'input-cat.png'}
(out/'request.json').write_text(json.dumps(saved_payload,indent=2))

def collect(request, chunks):
    return asyncio.run(collect_realtime_output(
        ws_url=f'ws://127.0.0.1:{args.port}/v1/realtime_video/generate',
        init_payload=request,events=[],num_chunks=chunks,require_chunk_stats=True))

with (out/'server.log').open('x') as log:
    server = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env=os.environ.copy())
    try:
        for _ in range(600):
            if server.poll() is not None:raise RuntimeError('Server exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=1) as r:
                    if r.status==200:break
            except (urllib.error.URLError,TimeoutError):pass
            time.sleep(1)
        else:raise TimeoutError('Native server was not ready after 600 seconds')
        warmup = collect(payload | {'max_chunks':2,'seed':41},2)
        (out/'warmup.json').write_text(json.dumps([asdict(s) for s in warmup.chunk_stats],indent=2))
        del warmup
        time.sleep(1)
        start = time.perf_counter()
        result = collect(payload,args.chunks)
        received_s = time.perf_counter()-start
        video = encode_realtime_frames_to_mp4(result.frames,fps=16)
        video_path = out/'output.mp4'
        video_path.write_bytes(video)
        saved_s = time.perf_counter()-start
        stats = [asdict(s) for s in result.chunk_stats]
        assert len(stats)==args.chunks
        assert [s['chunk_index'] for s in stats]==list(range(args.chunks))
        assert sum(s['num_frames'] for s in stats)==len(result.frames)
        assert all(f.shape==(480,832,3) for f in result.frames)
        frame_hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in result.frames]
        for index in sorted({0,len(result.frames)//3,2*len(result.frames)//3,len(result.frames)-1}):
            Image.fromarray(result.frames[index]).save(out/f'frame-{index:03d}.png')
        record = dict(label=args.label,source=(out/'source.txt').read_text().splitlines()[0],
            transport='native raw RGB WebSocket',warmup_chunks=2,chunks=args.chunks,
            frame_count=len(result.frames),fps=16,width=832,height=480,
            client_received_s=received_s,client_saved_s=saved_s,
            scheduler_forward_s=sum(s['scheduler_forward_ms'] for s in stats)/1000,
            chunk_total_s=sum(s['chunk_total_ms'] for s in stats)/1000,
            chunk_stats=stats,raw_frame_sha256=frame_hashes,
            output_path=str(video_path),output_sha256=hashlib.sha256(video).hexdigest())
        (out/'result.json').write_text(json.dumps(record,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k not in ('chunk_stats','raw_frame_sha256')},indent=2),flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait(timeout=10)
