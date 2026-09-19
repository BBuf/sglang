"""Repeated native Edge T2I/T2V validation of the committed combined candidate."""
import fcntl
import os
from pathlib import Path
import subprocess

root = Path('/campaign')
repo = root/'candidate-cosmos3-edge'
assert (root/'probe-cosmos3-t1.exit').read_text().strip() == '0'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'8',
    'PYTHONPATH':str(repo/'python'), 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def command(args, label, allow_disabled=False):
    assert not (root/f'cosmos3-edge-{label}.exit').exists()
    log_path = root/f'cosmos3-edge-{label}.log'
    with log_path.open('w') as log:
        p = subprocess.run(args, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/f'cosmos3-edge-{label}.exit').write_text(str(p.returncode))
    if p.returncode:
        assert allow_disabled and '[diffusion bcg] disabled' in log_path.read_text().lower(), label

def request(mode, source, label, flags=()):
    command(['python','-u',str(root/'run-cosmos3-edge.py'),'--mode',mode,
        '--repo',str(source),'--label',label,*flags], label, allow_disabled='--bcg' in flags)

with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    command(['python','-m','pytest','-q',
        'test/registered/kernels/ops/diffusion/test_cosmos3_edge_fusions.py',
        'python/sglang/multimodal_gen/test/unit/test_cosmos3.py'], 'combined-tests-retry1')
    command(['python','test/registered/kernels/benchmark/diffusion/bench_cosmos3_edge_fusions.py'], 'combined-marker-retry1')
for mode in ('t2i', 't2v'):
    for repeat in (1, 2):
        for arm in ('a1', 'b1', 'b2', 'a2'):
            source = root/'baseline' if arm.startswith('a') else repo
            label = f'combined-{mode}-r{repeat}-{arm}'
            request(mode, source, label)
    request(mode, repo, f'combined-{mode}-profile', ['--profile'])
for graph_mode in ('eager','bcg'):
    for arm in ('a1','b1','b2','a2'):
        quality = 'lossless' if arm.startswith('a') else 'high'
        flags = ['--quality', quality] + (['--bcg'] if graph_mode == 'bcg' else [])
        request('t2i', repo, f'quality-{graph_mode}-{arm}', flags)
for repeat in (1, 2):
    request('t2v', repo, f'quality-t2v-high-{repeat}', ['--quality','high'])
