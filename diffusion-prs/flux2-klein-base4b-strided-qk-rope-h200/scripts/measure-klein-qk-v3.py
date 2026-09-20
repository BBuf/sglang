"""Fixed two native ABBA groups for the NCU-driven pair-layout candidate."""
import json
from pathlib import Path
import re
import subprocess
root=Path('/campaign');repo=root/'candidate-klein-qk-pair-layout'
assert (root/'validate-klein-qk-v3.exit').read_text().strip()=='0'
baseline_repo=root/'baseline-super-current'
reference=json.loads((root/'artifacts/flux2-klein-base-4b/baseline-a1/result.json').read_text())
records=[]
for round in [1,2]:
 for arm,suffix in [('baseline','a1'),('candidate','b1'),('candidate','b2'),('baseline','a2')]:
  label=f'qk-v3-r{round}-{arm}-{suffix}'
  with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
   result=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--repo',str(baseline_repo if arm=='baseline' else repo),'--label',label],stdout=log,stderr=subprocess.STDOUT)
  (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(result.returncode));assert result.returncode==0,label
  row=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
  assert not row['error'] and row['output_sha256']==reference['output_sha256'],label
  log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'flux2-klein-base-4b-{label}.log').read_text())
  client=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log);assert len(client)==1
  item=dict(label=label,round=round,arm=arm,worker_s=row['e2e_latency_s'],denoise_s=row['denoise_latency_s'],client_saved_s=float(client[0]),output_sha256=row['output_sha256'])
  records.append(item);print(json.dumps(item),flush=True)
  (root/'artifacts/flux2-klein-base-4b/qk-abba-v3.json').write_text(json.dumps(records,indent=2))

label='qk-v3-candidate-profile'
with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
 result=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--repo',str(repo),'--label',label,'--profile'],stdout=log,stderr=subprocess.STDOUT)
(root/f'flux2-klein-base-4b-{label}.exit').write_text(str(result.returncode));assert result.returncode==0
row=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text());assert row['output_sha256']==reference['output_sha256']
trace=Path(row['native_profile_traces'][0]['path'])
with (trace.parent.parent/'inspect.txt').open('w') as out:
 subprocess.run(['python',str(root/'inspect-flux2-klein-trace.py'),str(trace)],stdout=out,check=True)
