"""Run existing numerical regressions and a separate CPU/compile diagnostic."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--head', required=True)
args = parser.parse_args()
root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
assert json.loads((root/'evidence/preflight/sana-video-ln-v1-paired1-admission.json').read_text())['passed']
folder = root/'block-runs/sana-video-ln-final-tests1'
folder.mkdir(exist_ok=False)
repo = '/task/candidates/sana-video-eager-ln-final'
container = 'bbuf-diffusion-rtx6000-20261026-sgl'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
base = ['docker', 'exec', '-e', 'FLASHINFER_DISABLE_VERSION_CHECK=1', '-e', 'PYTHONPATH='+repo+'/python', '-w', repo]
assert subprocess.check_output(base+[container, 'git', 'rev-parse', 'HEAD'], text=True).strip() == args.head
patch = subprocess.check_output(base+[container, 'git', 'diff', 'HEAD', '--binary', '--full-index'])
assert not patch
with (folder/'preflight.log').open('xb') as stream:
    subprocess.run(['docker', 'exec', '-e', 'PYTHONPATH=/task/repos/sglang/python', container, 'python3', '/task/preflight_ming_environment.py'], stdout=stream, stderr=subprocess.STDOUT, check=True)
commands = [
    ('numerical', base+['-e', 'CUDA_VISIBLE_DEVICES=0', container, 'python3', '-m', 'pytest', '-v',
        'test/registered/kernels/ops/diffusion/test_model_fast_paths.py',
        'test/registered/kernels/ops/diffusion/test_sites.py', '-k', 'sana_fused_ln_modulate_is_bit_exact or bitexact_gate']),
    ('cpu_compile', base+['-e', 'CUDA_VISIBLE_DEVICES=', container, 'python3', '/task/check_sana_ln_cpu_compile_fallback1.py']),
]
rows = []
for name, command in commands:
    started = time.monotonic()
    with (folder/(name+'.log')).open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    rows.append(dict(name=name, command=command, source_sha=args.head, source_diff_sha256=hashlib.sha256(patch).hexdigest(), returncode=result.returncode, seconds=time.monotonic()-started))
    (folder/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows[-1]), flush=True)
assert all(row['returncode'] == 0 for row in rows), 'Preserve test failures; no qualification admission'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
