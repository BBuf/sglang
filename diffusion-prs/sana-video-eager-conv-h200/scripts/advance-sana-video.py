"""Start official SANA-Video workload after Cosmos checkpoint cleanup."""
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root/'close-cosmos3-edge-cycle.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8',
    'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1','FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def command(args, label):
    assert not (root/f'sana-video-{label}.exit').exists()
    with (root/f'sana-video-{label}.log').open('w') as log:
        p = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/f'sana-video-{label}.exit').write_text(str(p.returncode))
    assert p.returncode == 0, label

command(['python','-u',str(root/'prepare-sana-video.py')], 'prepare')
for label, flags in [('baseline-eager-a1',[]),('baseline-eager-profile',['--profile']),
                     ('baseline-bcg-a1',['--bcg']),('baseline-bcg-profile',['--bcg','--profile'])]:
    command(['python','-u',str(root/'run-sana-video.py'),'--label',label,*flags], label)
