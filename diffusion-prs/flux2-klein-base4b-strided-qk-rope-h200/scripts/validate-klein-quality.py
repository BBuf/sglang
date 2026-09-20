"""Native quality and BCG applicability checks after the fixed eager comparison."""
import json
from pathlib import Path
import subprocess
import time
root=Path('/campaign'); done=root/'measure-klein-qk-v3.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
base=root/'baseline-super-current';candidate=root/'candidate-klein-qk-final'
for arm,repo in [('baseline',base),('candidate',candidate)]:
 for suffix,flags in [('high',['--quality','high']),('bcg-probe',['--bcg'])]:
  label=f'final-{arm}-{suffix}'
  with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
   proc=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--repo',str(repo),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
  (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(proc.returncode))
  row=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
  if suffix=='high':assert proc.returncode==0 and not row['error'],label
  else:assert row['error'] and not row['bcg_capture_detected'] and '[diffusion bcg] disabled' in row['bcg_invalid_signals'],label
  print(label,row['output_sha256'],flush=True)
