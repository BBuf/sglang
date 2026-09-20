"""Validate lossless kernels, collect paired NCU, then fixed native ABBA groups."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

root=Path('/campaign');repo=root/'candidate-klein-qk-tests'
assert (root/'screen-klein-qk-integration.exit').read_text().strip()=='0'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'klein-qk-tests-final.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),'5fd301db647e119757f14f707bb1e3c60c7a9316'],check=True)
with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1','OMP_NUM_THREADS':'8'}
 with (root/'test-klein-qk-v1.log').open('x') as log:
  result=subprocess.run(['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_flux2_strided_qknorm_rope.py'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 (root/'test-klein-qk-v1.exit').write_text(str(result.returncode));assert result.returncode==0

ncu='/opt/nvidia/nsight-compute/2025.3.1/ncu';profile=root/'profile/klein-qk-rope-h200';profile.mkdir(exist_ok=False)
for arm in ['baseline','candidate']:
 with (profile/f'{arm}-ncu.log').open('x') as log:
  result=subprocess.run([ncu,'--set','full','--target-processes','all','--nvtx','--nvtx-include','profile/','--force-overwrite','-o',str(profile/arm),'python',str(root/'bench-klein-qk.py'),'--ncu',arm],stdout=log,stderr=subprocess.STDOUT)
 (profile/f'{arm}-ncu.exit').write_text(str(result.returncode));assert result.returncode==0
 with (profile/f'{arm}-raw.csv').open('w') as out:
  subprocess.run([ncu,'--import',str(profile/f'{arm}.ncu-rep'),'--page','raw','--csv'],stdout=out,check=True)
 print('NCU',arm,'complete',flush=True)

baseline_repo=root/'baseline-super-current'
reference=json.loads((root/'artifacts/flux2-klein-base-4b/baseline-a1/result.json').read_text())
records=[]
for round in [1,2]:
 for arm,suffix in [('baseline','a1'),('candidate','b1'),('candidate','b2'),('baseline','a2')]:
  label=f'qk-r{round}-{arm}-{suffix}'
  with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
   result=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--repo',str(baseline_repo if arm=='baseline' else repo),'--label',label],stdout=log,stderr=subprocess.STDOUT)
  (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(result.returncode));assert result.returncode==0,label
  row=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
  assert not row['error'] and row['output_sha256']==reference['output_sha256'],label
  log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'flux2-klein-base-4b-{label}.log').read_text())
  client=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log);assert len(client)==1
  item=dict(label=label,round=round,arm=arm,worker_s=row['e2e_latency_s'],denoise_s=row['denoise_latency_s'],client_saved_s=float(client[0]),output_sha256=row['output_sha256'])
  records.append(item);print(json.dumps(item),flush=True)
  (root/'artifacts/flux2-klein-base-4b/qk-abba-v1.json').write_text(json.dumps(records,indent=2))
