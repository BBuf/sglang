"""After fixed v2 ABBA completes, test the NCU-driven register-pair edit."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

root=Path('/campaign');done=root/'validate-klein-qk-v2.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
repo=root/'candidate-klein-qk-pair-layout'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'klein-qk-pair-layout.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),'454ee669a40c4119967a0909449237dfbdc63998'],check=True)
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1','OMP_NUM_THREADS':'8'}
 with (root/'test-klein-qk-v3.log').open('x') as log:
  result=subprocess.run(['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_flux2_strided_qknorm_rope.py'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 (root/'test-klein-qk-v3.exit').write_text(str(result.returncode));assert result.returncode==0
with (root/'bench-klein-qk-v3.log').open('x') as log:
 result=subprocess.run(['python','-u',str(root/'bench-klein-qk-v3.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'bench-klein-qk-v3.exit').write_text(str(result.returncode));assert result.returncode==0
ncu='/opt/nvidia/nsight-compute/2025.3.1/ncu';profile=root/'profile/klein-qk-rope-v3-h200';profile.mkdir(exist_ok=False)
with (profile/'candidate-ncu.log').open('x') as log:
 result=subprocess.run([ncu,'--set','full','--target-processes','all','--nvtx','--nvtx-include','profile/','--force-overwrite','-o',str(profile/'candidate'),'python',str(root/'bench-klein-qk-v3.py'),'--ncu','candidate'],stdout=log,stderr=subprocess.STDOUT)
(profile/'candidate-ncu.exit').write_text(str(result.returncode));assert result.returncode==0
print('Pair-layout correctness, focused benchmark and NCU complete',flush=True)
