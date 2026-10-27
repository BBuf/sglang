"""Generate frozen Joy edit inputs with native MOSS main; not performance evidence."""
import base64
import hashlib
import json
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
CONTAINER = 'bbuf-diffusion-rtx6000-20261026-sgl'
RUN = 'joy-aspect-reference-generation1'
folder = ROOT / 'block-runs' / RUN
folder.mkdir(exist_ok=False)
repo = '/task/repos/sglang'


def command(args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    return not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


assert idle()
assert command(['docker', 'exec', CONTAINER, 'git', '-C', repo, 'rev-parse', 'HEAD']) == 'af21e2b989e482cfdce39689126562f2209d53ac'
assert not command(['docker', 'exec', CONTAINER, 'git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
manifest = ROOT / 'evidence/downloads/moss-modelscope-download-manifest.json'
weights = json.loads(manifest.read_text())
assert weights['complete'] and all(f['verified_against_hf'] for f in weights['files'])
contract = json.loads((ROOT / 'inputs/moss-workloads.json').read_text())
metadata = dict(purpose='Benchmark reference-image generation, not quality oracle or performance evidence',
                source_sha='af21e2b989e482cfdce39689126562f2209d53ac', weights_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                reference_policy='Keep the first result for each prospectively chosen shape; no rerolls or sample selection.',
                shapes=[[1792,512],[512,1792]], results=[])
pid_path = f'/task/block-runs/{RUN}/server.pid'
launch = ['docker','exec',CONTAINER,'sh','-c',f'echo $$ > {pid_path}; exec sh /task/launch_sgl_moss_cudnn.sh']
metadata['launch_command'] = launch
(folder / 'launch.sh').write_bytes((ROOT / 'launch_sgl_moss_cudnn.sh').read_bytes())
log = folder / 'server.log'
start = time.monotonic()
with log.open('wb') as stream:
    server = subprocess.Popen(launch, stdout=stream, stderr=subprocess.STDOUT)
try:
    while True:
        assert server.poll() is None, 'Reference generator server exited'
        assert time.monotonic()-start < 1200, 'Readiness timeout'
        try:
            with urllib.request.urlopen('http://127.0.0.1:31030/liveness', timeout=2) as response:
                ready = response.status == 200
        except Exception:
            ready = False
        if ready and b'Warmup requests: 100%' in log.read_bytes():
            break
        time.sleep(2)
    metadata['launch_to_ready_seconds'] = time.monotonic()-start
    for width, height in metadata['shapes']:
        name = f'reference-teapot-{width}x{height}.png'
        image = ROOT / 'inputs' / name
        assert not image.exists()
        target = folder / f'{width}x{height}'
        target.mkdir()
        payload = {k:contract[k] for k in ['seed','num_inference_steps','guidance_scale','flow_shift','negative_prompt','generator_device','uncond_skip_layers','slg_start','slg_stop']}
        payload.update(model='/task/cache/common/MOSS-Image-Base', prompt=contract['prompts'][0]['prompt'],
                       size=f'{width}x{height}', n=1, response_format='b64_json', output_format='png', quality='lossless')
        body = json.dumps(payload).encode()
        (target / 'request.json').write_bytes(body)
        request = urllib.request.Request('http://127.0.0.1:31030/v1/images/generations', data=body, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=1200) as response:
            raw = response.read()
        (target / 'response.json').write_bytes(raw)
        data = json.loads(raw)['data']
        assert len(data) == 1
        content = base64.b64decode(data[0]['b64_json'])
        assert content[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', content[16:24]) == (width,height)
        (target / 'output.png').write_bytes(content)
        image.write_bytes(content)
        metadata['results'].append(dict(path='inputs/'+name, sha256=hashlib.sha256(content).hexdigest(), width=width, height=height, benchmark=False))
        print(json.dumps(metadata['results'][-1]), flush=True)
        (folder / 'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    metadata['status'] = 'completed'
except Exception as error:
    metadata.update(status='failed', error=repr(error))
    raise
finally:
    (folder / 'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    pid_file = folder / 'server.pid'
    if server.poll() is None and pid_file.exists():
        subprocess.run(['docker','exec',CONTAINER,'kill','-TERM',pid_file.read_text().strip()],check=True)
    server.wait(timeout=60)
    for _ in range(30):
        if idle():
            break
        time.sleep(1)
    assert idle()
