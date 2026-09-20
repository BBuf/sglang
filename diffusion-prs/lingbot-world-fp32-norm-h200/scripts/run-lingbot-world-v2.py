"""Pinned native LingBot World V2 saved-output benchmark and diagnostic request."""
import argparse
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--repo',default='/campaign/baseline')
p.add_argument('--label',required=True)
p.add_argument('--gpus',type=int,choices=(1,2),default=1)
p.add_argument('--frames',type=int,choices=(9,45,117),default=9)
p.add_argument('--profile',action='store_true')
p.add_argument('--all-stages',action='store_true')
p.add_argument('--bcg',action='store_true')
p.add_argument('--quality',choices=('lossless','high'),default='lossless')
args=p.parse_args()
root=Path('/campaign')
lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
repo=Path(args.repo).resolve()
os.environ.update(json.loads((root/'artifacts/lingbot-world-v2-cache-env.json').read_text()))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(repo/'python'),
    OMP_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='0' if args.gpus==1 else '0,1',SGLANG_DIFFUSION_SYNC_STAGE_PROFILING='1')
sys.path.insert(0,str(repo/'python'));os.chdir(repo)
import sglang
import torch
assert Path(sglang.__file__).resolve().is_relative_to(repo)
assert torch.cuda.device_count()==args.gpus
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if int(line.split(',')[0].strip()) < args.gpus}
for _ in range(30):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):
        break
    time.sleep(1)
else:
    raise RuntimeError('Assigned GPU remains busy; no process was killed')
subprocess.run(['git','diff','--quiet'],check=True)
subprocess.run(['git','diff','--cached','--quiet'],check=True)
out=root/'artifacts/lingbot-world-v2'/args.label
out.mkdir(parents=True,exist_ok=False)
(out/'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True)+str(sglang.__file__)+'\n')
(out/'gpu-before.txt').write_text(inventory+'\nProcesses:\n'+processes)
helper=repo/'python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
key='lingbot-world-v2'
preset=bench.MODELS[key]
preset['image_path']=str(root/'artifacts/input-media/longlive2-cat.png')
if args.frames != 9:
    preset['extra_args'] = [a for a in preset['extra_args'] if not a.startswith('--num-frames=')]
    preset['extra_args'].append('--num-frames='+str(args.frames))
    preset['config_overrides']['actions'] = [['w'] for _ in range(args.frames)]
preset['extra_args'] += ['--num-gpus='+str(args.gpus),'--tp-size=1',
    '--ulysses-degree='+str(args.gpus),'--enable-torch-compile=false',
    '--performance-mode=manual','--dit-layerwise-offload=false','--dit-cpu-offload=false',
    '--vae-cpu-offload=false','--warmup-resolutions=832x480','--warmup-num-frames=9']
if args.profile:
    # The native causal loop records StageProfiler metrics but may not
    # call SGLDiffusionProfiler.step_denoising_step. Use the supported full
    # request profiler, then slice complete causal blocks offline.
    preset['extra_args'] += ['--profile','--profile-all-stages']
    os.environ['SGLANG_DIFFUSION_TORCH_PROFILER_DIR']=str(out/'traces')
bench.MODELS[key]=preset
(out/'preset.json').write_text(json.dumps(preset,indent=2))
result=bench._run_benchmark_once_impl(key,args.label,out,warmup=True,torch_compile=False,
    quality=args.quality,breakable_cuda_graph=args.bcg,
    model_cache_dir=root/'model-caches/lingbot-world-v2-cycle',cuda_visible_devices='0' if args.gpus==1 else '0,1')
if args.profile and not result.get('error'):
    traces=sorted((out/'traces').rglob('*.trace.json.gz'))
    if not traces:
        result.update(error=True,error_reason='Profile requested but no native trace was produced')
    else:
        evidence=[]
        for trace in traces:
            with gzip.open(trace,'rt') as f:events=json.load(f)['traceEvents']
            kernels=sum(e.get('cat')=='kernel' for e in events)
            assert kernels>0,trace
            evidence.append(dict(path=str(trace),event_count=len(events),kernel_count=kernels))
        result['native_profile_traces']=evidence
(out/'result.json').write_text(json.dumps(result,indent=2))
raise SystemExit(int(bool(result.get('error'))))
