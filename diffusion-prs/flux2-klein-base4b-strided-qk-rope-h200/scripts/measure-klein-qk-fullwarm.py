"""Fixed two native ABBA groups with a full 50-step same-shape request warmup in both arms; initial one-step-warmup measurements are retained separately."""
import json
from pathlib import Path
import re
import subprocess
import time
root=Path('/campaign');repo=root/'candidate-klein-qk-final'
done=root/'validate-klein-quality.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
baseline_repo=root/'baseline-super-current'
reference=json.loads((root/'artifacts/flux2-klein-base-4b/baseline-a1/result.json').read_text())
records=[]
for round in [1,2]:
 for arm,suffix in [('baseline','a1'),('candidate','b1'),('candidate','b2'),('baseline','a2')]:
  label=f'qk-fullwarm-r{round}-{arm}-{suffix}'
  with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
   result=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b-fullwarm.py'),'--repo',str(baseline_repo if arm=='baseline' else repo),'--label',label,'--warmup-steps','50'],stdout=log,stderr=subprocess.STDOUT)
  (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(result.returncode));assert result.returncode==0,label
  row=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
  assert not row['error'] and row['output_sha256']==reference['output_sha256'],label
  log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'flux2-klein-base-4b-{label}.log').read_text())
  client=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log);assert len(client)==1
  item=dict(label=label,round=round,arm=arm,worker_s=row['e2e_latency_s'],denoise_s=row['denoise_latency_s'],client_saved_s=float(client[0]),output_sha256=row['output_sha256'])
  records.append(item);print(json.dumps(item),flush=True)
  (root/'artifacts/flux2-klein-base-4b/qk-abba-fullwarm.json').write_text(json.dumps(records,indent=2))
