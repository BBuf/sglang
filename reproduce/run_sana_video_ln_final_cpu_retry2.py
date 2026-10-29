"""Keep passing GPU tests; retry CPU tensors with CUDA visible for module imports."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
head = 'e0d694593a65cb5b463c4011b84ee50abefa5e43'
container = 'bbuf-diffusion-rtx6000-20261026-sgl'
repo = '/task/candidates/sana-video-eager-ln-final'
old = root/'block-runs/sana-video-ln-final-tests1/results.json'
previous = json.loads(old.read_text())
assert len(previous) == 2 and previous[0]['name'] == 'numerical' and previous[0]['returncode'] == 0
assert previous[1]['name'] == 'cpu_compile' and previous[1]['returncode'] == 1
assert all(row['source_sha'] == head for row in previous)
failure_log = old.parent/'cpu_compile.log'
assert 'RuntimeError: No CUDA GPUs are available' in failure_log.read_text()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() == head
assert not subprocess.check_output(['docker', 'exec', container, 'git', '-C', repo, 'diff', 'HEAD', '--binary', '--full-index'])
folder = root/'block-runs/sana-video-ln-final-tests2'
folder.mkdir(exist_ok=False)
command = ['docker', 'exec', '-e', 'FLASHINFER_DISABLE_VERSION_CHECK=1', '-e', 'CUDA_VISIBLE_DEVICES=0',
           '-e', 'PYTHONPATH='+repo+'/python', '-w', repo, container, 'python3', '/task/check_sana_ln_cpu_compile_fallback1.py']
started = time.monotonic()
with (folder/'cpu_compile.log').open('xb') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
retry = dict(name='cpu_compile', command=command, source_sha=head,
             source_diff_sha256=hashlib.sha256(b'').hexdigest(), returncode=result.returncode,
             seconds=time.monotonic()-started,
             correction='CUDA visible for existing import-time Triton device query; diagnostic tensors and FX graph remain CPU. No model code or diagnostic script changed.',
             original_failed_result=str(old), original_failed_log_sha256=hashlib.sha256(failure_log.read_bytes()).hexdigest())
rows = [dict(previous[0], reused_passing_result=str(old)), retry]
(folder/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(retry), flush=True)
assert result.returncode == 0, 'Preserve failed retry; do not run compatibility'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
