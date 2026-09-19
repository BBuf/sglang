"""Validate the committed eager convolution candidate with native saved requests."""
import fcntl
import os
from pathlib import Path
import subprocess

root = Path('/campaign')
repo = root/'candidate-sana-video'
assert (root/'advance-sana-video.exit').read_text().strip() == '0'
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == '275113434ffe34412f807514399ba00ef70fa38e'
env = os.environ | {'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'8',
    'PYTHONPATH':str(repo/'python'), 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING':'1',
    'FLASHINFER_DISABLE_VERSION_CHECK':'1'}

def command(args, label):
    assert not (root/f'sana-video-{label}.exit').exists()
    with (root/f'sana-video-{label}.log').open('w') as log:
        process = subprocess.run(args, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/f'sana-video-{label}.exit').write_text(str(process.returncode))
    assert process.returncode == 0, label

with (root/'gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    command(['python','-m','pytest','-q',
        'test/registered/kernels/ops/diffusion/test_sana_video_eager_conv.py',
        'python/sglang/multimodal_gen/test/unit/test_sana_video.py'], 'conv-tests')
    command(['python','test/registered/kernels/benchmark/diffusion/bench_sana_video_eager_conv.py'], 'conv-marker')

for mode, repeats in [('eager', (1,2)), ('bcg', (1,))]:
    for repeat in repeats:
        for arm in ('a1','b1','b2','a2'):
            source = root/'baseline' if arm.startswith('a') else repo
            command(['python','-u',str(root/'run-sana-video.py'),
                '--repo',str(source),'--label',f'conv-{mode}-r{repeat}-{arm}',
                *(['--bcg'] if mode=='bcg' else [])], f'conv-{mode}-r{repeat}-{arm}')
    command(['python','-u',str(root/'run-sana-video.py'),
        '--repo',str(repo),'--label',f'conv-{mode}-profile','--profile',
        *(['--bcg'] if mode=='bcg' else [])], f'conv-{mode}-profile')
