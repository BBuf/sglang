"""Validate the copy-only candidate before native saved-output screening."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/campaign');repo=root/'candidate-joy-qkv-cat-v1'
assert (root/'advance-joy-image-edit-retry1.exit').read_text().strip()=='0'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'joy-qkv-cat-v1.bundle'),'HEAD'],check=True)
head=subprocess.check_output(['git','-C',str(root/'baseline-super-current'),'rev-parse','FETCH_HEAD'],text=True).strip()
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),head],check=True)
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1','OMP_NUM_THREADS':'8'}
 with (root/'test-joy-qkv-cat-v1.log').open('x') as log:
  proc=subprocess.run(['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_joint_qkv_cat.py'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 (root/'test-joy-qkv-cat-v1.exit').write_text(str(proc.returncode));assert proc.returncode==0
with (root/'bench-joy-qkv-cat-v1.log').open('x') as log:
 proc=subprocess.run(['python','-u',str(root/'bench-joy-qkv-cat-v1.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'bench-joy-qkv-cat-v1.exit').write_text(str(proc.returncode));assert proc.returncode==0
reference=json.loads((root/'artifacts/joy-image-edit/baseline-retry1/result.json').read_text())
for label,flags in [('qkv-cat-candidate-b0',[]),('qkv-cat-candidate-profile',['--profile'])]:
 with (root/f'joy-image-edit-{label}.log').open('x') as log:
  proc=subprocess.run(['python','-u',str(root/'run-joy-image-edit-retry1.py'),'--repo',str(repo),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'joy-image-edit-{label}.exit').write_text(str(proc.returncode));assert proc.returncode==0,label
 data=json.loads((root/'artifacts/joy-image-edit'/label/'result.json').read_text())
 assert not data['error'] and data['output_sha256']==reference['output_sha256'],label
 print(label,json.dumps(data),flush=True)
 if flags:
  trace=Path(next(row['path'] for row in data['native_profile_traces'] if 'global-rank0' in row['path']))
  with (trace.parent.parent/'inspect.txt').open('w') as out:
   subprocess.run(['python',str(root/'inspect-joy-image-edit-trace.py'),str(trace)],stdout=out,check=True)
  report=json.loads((trace.parent/'forward3-evidence.json').read_text())
  fused=[r for r in report['top_kernels'] if r['name']=='_joint_qkv_cat_kernel']
  assert len(fused)==1 and fused[0]['count']==40,fused
  print('Forty fused joint QKV concatenations in complete native forward',fused,flush=True)
