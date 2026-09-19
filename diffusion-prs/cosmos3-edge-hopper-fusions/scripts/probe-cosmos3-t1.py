"""Validate Edge GQA rounding before native saved-output A/B and profiling."""
import fcntl
import os
from pathlib import Path
import subprocess

root = Path('/campaign')
repo = root/'candidate-cosmos3-edge'
assert (root/'advance-cosmos3-edge-retry1.exit').read_text().strip() == '0'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'8',
    'PYTHONPATH':str(repo/'python'), 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def command(args, label):
    assert not (root/f'cosmos3-edge-{label}.exit').exists()
    with (root/f'cosmos3-edge-{label}.log').open('w') as log:
        p = subprocess.run(args, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/f'cosmos3-edge-{label}.exit').write_text(str(p.returncode))
    assert p.returncode == 0, label

with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    command(['python','-m','pytest','-q',
        'test/registered/kernels/ops/diffusion/test_cosmos3_edge_qk_rope.py',
        'python/sglang/multimodal_gen/test/unit/test_cosmos3.py::TestCosmos3T1FusedQKNormRoPE'], 't1-tests')
for arm in ('a1', 'b1', 'b2', 'a2'):
    source = root/'baseline' if arm.startswith('a') else repo
    label = 't1-r1-'+arm
    command(['python','-u',str(root/'run-cosmos3-edge.py'),'--mode','t2i',
        '--repo',str(source),'--label',label], label)
command(['python','-u',str(root/'run-cosmos3-edge.py'),'--mode','t2i',
    '--repo',str(repo),'--label','t1-candidate-profile','--profile'], 't1-candidate-profile')
