"""Admit Cosmos3 Edge after LongCat cleanup and collect native baselines."""
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root/'close-longcat-image-cycle-retry1.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'8',
    'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1', 'FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def command(args, label, allow_disabled=False):
    log_path = root/f'cosmos3-edge-{label}.log'
    assert not (root/f'cosmos3-edge-{label}.exit').exists()
    with log_path.open('w') as log:
        p = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/f'cosmos3-edge-{label}.exit').write_text(str(p.returncode))
    if p.returncode:
        assert allow_disabled and '[diffusion bcg] disabled' in log_path.read_text().lower(), log_path

command(['python', '-u', str(root/'prepare-cosmos3-edge.py')], 'prepare')
for mode in ('t2i', 't2v'):
    for suffix, flags in [('baseline-a1', []), ('baseline-profile', ['--profile']),
                          ('bcg-applicability', ['--bcg'])]:
        label = f'{mode}-{suffix}'
        command(['python', '-u', str(root/'run-cosmos3-edge.py'), '--mode', mode,
                 '--label', label, *flags], label, allow_disabled=suffix == 'bcg-applicability')
