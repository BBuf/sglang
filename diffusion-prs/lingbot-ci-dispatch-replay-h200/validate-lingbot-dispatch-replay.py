"""Validate the remaining live-dispatch test assumptions after Cosmos timings."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

root=Path('/campaign');done=root/'validate-cosmos3-super-replication-retry1.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
repo=root/'candidate-lingbot-dispatch-replay'
subprocess.run(['git','-C',str(root/'baseline-current'),'fetch',str(root/'lingbot-ci-dispatch-replay.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-current'),'worktree','add','--detach',str(repo),'e0b81e1f947872a10fc6ad2dec349cd0222b50e6'],check=True)
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 for _ in range(35):
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if not any(line.startswith(uuid) for line in processes.splitlines()):break
  time.sleep(1)
 else:raise RuntimeError('Assigned GPU busy; no process killed')
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'OMP_NUM_THREADS':'8','FLASHINFER_DISABLE_VERSION_CHECK':'1'}
 result=subprocess.run(['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_lingbot_modulation.py'],cwd=repo,env=env)
 assert result.returncode==0
