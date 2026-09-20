"""Validate the final portability guards and repeat full-warm ABBA on the final commit."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root=Path('/campaign');repo=root/'candidate-joy-qkv-cat-final';art=root/'artifacts/joy-image-edit'
done=root/'validate-joy-qkv-cat-v2.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'joy-qkv-cat-final.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),'3b00154ddb33ac436906a493c9706ed3855ec918'],check=True)

def command(name,cmd,**kwargs):
 with (root/f'{name}.log').open('x') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,**kwargs)
 (root/f'{name}.exit').write_text(str(result.returncode))
 return result.returncode

with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1','OMP_NUM_THREADS':'8'}
 assert command('test-joy-qkv-cat-final',['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_joint_qkv_cat.py'],cwd=repo,env=env)==0
reference=json.loads((art/'baseline-retry1/result.json').read_text())
records=[]
for round in [1,2]:
 for arm,suffix in [('baseline','a1'),('candidate','b1'),('candidate','b2'),('baseline','a2')]:
  label=f'qkv-final-fullwarm-r{round}-{arm}-{suffix}'
  code=command(f'joy-image-edit-{label}',['python','-u',str(root/'run-joy-image-edit-fullwarm.py'),
   '--repo',str(root/'baseline-super-current' if arm=='baseline' else repo),'--label',label,'--warmup-steps','40'])
  assert code==0,label
  row=json.loads((art/label/'result.json').read_text())
  assert not row['error'] and row['output_sha256']==reference['output_sha256'],label
  log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'joy-image-edit-{label}.log').read_text())
  clients=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log);assert len(clients)==1
  item=dict(label=label,round=round,arm=arm,worker_s=row['e2e_latency_s'],denoise_s=row['denoise_latency_s'],
   client_saved_s=float(clients[0]),peak_gib=row['peak_memory_gb'],output_sha256=row['output_sha256'])
  records.append(item);print(json.dumps(item),flush=True)
  (art/'qkv-abba-fullwarm-final.json').write_text(json.dumps(records,indent=2))
(root/'measure-joy-qkv-cat-final.exit').write_text('0')

for label,which,flags in [
 ('qkv-final-candidate-profile',repo,['--profile']),
 ('qkv-final-candidate-high',repo,['--quality','high']),
 ('qkv-final-candidate-bcg',repo,['--bcg']),
 ('qkv-final-candidate-high-bcg',repo,['--quality','high','--bcg']),
]:
 code=command(f'joy-image-edit-{label}',['python','-u',str(root/'run-joy-image-edit-retry1.py'),
  '--repo',str(which),'--label',label,*flags])
 row=json.loads((art/label/'result.json').read_text())
 if '--bcg' in flags:
  assert code!=0 and row['error'] and row.get('bcg_invalid_signals'),label
 else:
  assert code==0 and not row['error'],label
  if '--quality' not in flags:assert row['output_sha256']==reference['output_sha256'],label
 if '--profile' in flags:
  trace=Path(next(x['path'] for x in row['native_profile_traces'] if 'global-rank0' in x['path']))
  with (trace.parent.parent/'inspect.txt').open('w') as out:
   subprocess.run(['python',str(root/'inspect-joy-image-edit-trace.py'),str(trace)],stdout=out,check=True)
 print('Validated',label,json.dumps(row),flush=True)
assert json.loads((art/'qkv-v2-baseline-high/result.json').read_text())['output_sha256']==json.loads((art/'qkv-final-candidate-high/result.json').read_text())['output_sha256']
print('Fixed ABBA, profile and same-mode quality/applicability evidence complete',flush=True)
