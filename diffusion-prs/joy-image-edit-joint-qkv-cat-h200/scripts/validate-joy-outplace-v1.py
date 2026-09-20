"""Validate existing out-of-place image QK/RoPE plus joint copies with fixed full-warm ABBA."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root=Path('/campaign');repo=root/'candidate-joy-outplace-v1';art=root/'artifacts/joy-image-edit'
done=root/'probe-joy-outplace-qk.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
subprocess.run(['git','-C',str(root/'baseline-super-current'),'fetch',str(root/'joy-outplace-v1.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(root/'baseline-super-current'),'worktree','add','--detach',str(repo),'4811f4d52aa2586412f699b3bb84ed184d76250a'],check=True)

def command(name,cmd,**kwargs):
 with (root/f'{name}.log').open('x') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,**kwargs)
 (root/f'{name}.exit').write_text(str(result.returncode))
 return result.returncode

with (root/'gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(repo/'python'),'FLASHINFER_DISABLE_VERSION_CHECK':'1','OMP_NUM_THREADS':'8'}
 assert command('test-joy-outplace-v1',['python','-m','pytest','-q','test/registered/kernels/ops/diffusion/test_joint_qkv_cat.py','test/registered/kernels/ops/diffusion/test_joy_strided_qk_rope.py'],cwd=repo,env=env)==0
assert command('bench-joy-outplace-v1',['python','-u',str(root/'bench-joy-outplace-v1.py')])==0

profile=root/'profile/joy-outplace-v1-h200';profile.mkdir(exist_ok=False)
for child in ['harness','reports','analysis']:(profile/child).mkdir()
shutil.copy2(root/'bench-joy-outplace-v1.py',profile/'harness/bench-joy-outplace-v1.py')
ncu='/opt/nvidia/nsight-compute/2025.3.1/ncu'
for arm in ['baseline','candidate']:
 args=[ncu,'--set','full','--section','PmSampling','--target-processes','all','--nvtx','--nvtx-include','profile/',
  '-o',str(profile/'reports'/arm),'python',str(profile/'harness/bench-joy-outplace-v1.py'),'--ncu',arm]
 assert command(f'ncu-joy-outplace-v1-{arm}',args)==0
assert command('ncu-joy-outplace-v1-candidate-source',[ncu,'--set','source','--section','SourceCounters',
 '--target-processes','all','--nvtx','--nvtx-include','profile/','-o',str(profile/'reports/candidate-source'),
 'python',str(profile/'harness/bench-joy-outplace-v1.py'),'--ncu','candidate'])==0
(root/'ncu-joy-outplace-v1.exit').write_text('0')

reference=json.loads((art/'baseline-retry1/result.json').read_text())
records=[]
for round in [1,2]:
 for arm,suffix in [('baseline','a1'),('candidate','b1'),('candidate','b2'),('baseline','a2')]:
  label=f'outplace-fullwarm-r{round}-{arm}-{suffix}'
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
  (art/'qkv-abba-fullwarm-outplace-v1.json').write_text(json.dumps(records,indent=2))
(root/'measure-joy-outplace-v1.exit').write_text('0')

for label,which,flags in [
 ('outplace-v1-candidate-profile',repo,['--profile']),
 ('outplace-v1-baseline-high',root/'baseline-super-current',['--quality','high']),
 ('outplace-v1-candidate-high',repo,['--quality','high']),
 ('outplace-v1-candidate-bcg',repo,['--bcg']),
 ('outplace-v1-baseline-high-bcg',root/'baseline-super-current',['--quality','high','--bcg']),
 ('outplace-v1-candidate-high-bcg',repo,['--quality','high','--bcg']),
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
assert json.loads((art/'outplace-v1-baseline-high/result.json').read_text())['output_sha256']==json.loads((art/'outplace-v1-candidate-high/result.json').read_text())['output_sha256']
print('Fixed ABBA, profile and same-mode quality/applicability evidence complete',flush=True)
