"""Admit a profile-motivated Hopper TP2 fusion against a fresh current-main baseline."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/campaign')
base=root/'baseline-super-current';candidate=root/'candidate-cosmos3-super-t2i'
subprocess.run(['git','-C',str(root/'baseline-current'),'fetch',str(root/'cosmos3-super-prototype.bundle'),'HEAD'],check=True)
for repo,head in [(base,'80da4432d085ed4d6166ef643d9fd2b829dbb0c5'),(candidate,'86c40bd29225f21f986aff26101f1c47f58695b8')]:
 subprocess.run(['git','-C',str(root/'baseline-current'),'worktree','add','--detach',str(repo),head],check=True)
env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','PYTHONPATH':str(candidate/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1'}
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 selected=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 for _ in range(35):
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if not any(line.startswith(selected) for line in procs.splitlines()):break
  time.sleep(1)
 else:raise RuntimeError('Assigned GPU busy; no process killed')
 for label,args in [('model-tests',['python/sglang/multimodal_gen/test/unit/test_cosmos3.py']),('existing-kernel-tests',['test/registered/kernels/ops/diffusion/test_rope.py','-k','qknorm_rope_pack_kv'])]:
  with (root/f'cosmos3-super-prototype-{label}.log').open('x') as log:
   r=subprocess.run(['python','-m','pytest','-q',*args],cwd=candidate,env=env,stdout=log,stderr=subprocess.STDOUT)
  (root/f'cosmos3-super-prototype-{label}.exit').write_text(str(r.returncode));assert r.returncode==0,label
rows=[]
for label,repo,flags in [('current-baseline-a1',base,[]),('prototype-candidate-a1',candidate,[]),('prototype-profile',candidate,['--profile','--all-stages'])]:
 with (root/f'cosmos3-super-t2i-{label}.log').open('x') as log:
  r=subprocess.run(['python','-u',str(root/'run-cosmos3-super-t2i.py'),'--repo',str(repo),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'cosmos3-super-t2i-{label}.exit').write_text(str(r.returncode));assert r.returncode==0,label
 result=json.loads((root/'artifacts/cosmos3-super-t2i'/label/'result.json').read_text())
 assert result['output_sha256']==['cfd2735b1e5d3b64a888ccc631f48d173e95c32857e8cbebd9e80fde3bc90d5e'],label
 rows.append(result)
print(json.dumps(rows,indent=2))
