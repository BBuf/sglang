"""Admit the next public checkpoint only after the preceding model is closed."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/campaign');done=root/'close-cosmos3-super-i2v-cycle.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
assert not (root/'model-caches/cosmos3-super-i2v-cycle').exists()
with (root/'prepare-flux2-klein-base-4b.log').open('x') as log:
 result=subprocess.run(['python','-u',str(root/'prepare-flux2-klein-base-4b.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'prepare-flux2-klein-base-4b.exit').write_text(str(result.returncode));assert result.returncode==0
for label,flags in [('baseline-a1',[]),('baseline-profile',['--profile']),('baseline-bcg-probe',['--bcg'])]:
 with (root/f'flux2-klein-base-4b-{label}.log').open('x') as log:
  result=subprocess.run(['python','-u',str(root/'run-flux2-klein-base-4b.py'),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'flux2-klein-base-4b-{label}.exit').write_text(str(result.returncode))
 data=json.loads((root/'artifacts/flux2-klein-base-4b'/label/'result.json').read_text())
 if label=='baseline-bcg-probe' and result.returncode:
  assert data['error'] and data.get('bcg_invalid_signals'),data
 else:assert result.returncode==0 and not data['error'],label
print('Native base-4B admission/profile and actual BCG applicability recorded')
