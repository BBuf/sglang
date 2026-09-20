"""Validate Hopper fusion and simulated non-Hopper native replay dispatch."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
repo = root / 'candidate-lingbot-hopper-guard'
subprocess.run(['git', '-C', str(root / 'baseline-current'), 'fetch',
                str(root / 'lingbot-hopper-guard.bundle'), 'HEAD'], check=True)
subprocess.run(['git', '-C', str(root / 'baseline-current'), 'worktree', 'add',
                '--detach', str(repo), '0267caed8ab9872f401b65d586c73fa559d514fc'], check=True)
with (root / 'gpu.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    for _ in range(35):
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if not any(line.startswith(uuid) for line in processes.splitlines()):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Assigned GPU busy; no process killed')
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': str(repo / 'python'),
                        'OMP_NUM_THREADS': '8', 'FLASHINFER_DISABLE_VERSION_CHECK': '1'}
    result = subprocess.run(['python', '-m', 'pytest', '-q',
                             'test/registered/kernels/ops/diffusion/test_lingbot_modulation.py'], cwd=repo, env=env)
    assert result.returncode == 0
