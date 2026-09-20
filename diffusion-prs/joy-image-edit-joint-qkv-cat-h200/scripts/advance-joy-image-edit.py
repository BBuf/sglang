"""Admit the next public checkpoint only after the preceding model is closed."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/campaign');done=root/'close-flux2-klein-base-4b-cycle.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
assert not (root/'model-caches/flux2-klein-base-4b-cycle').exists()
with (root/'prepare-joy-image-edit.log').open('x') as log:
 result=subprocess.run(['python','-u',str(root/'prepare-joy-image-edit.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'prepare-joy-image-edit.exit').write_text(str(result.returncode));assert result.returncode==0
for label,flags in [('baseline-a1',[]),('baseline-profile',['--profile']),('baseline-bcg-probe',['--bcg'])]:
 with (root/f'joy-image-edit-{label}.log').open('x') as log:
  result=subprocess.run(['python','-u',str(root/'run-joy-image-edit.py'),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'joy-image-edit-{label}.exit').write_text(str(result.returncode))
 data=json.loads((root/'artifacts/joy-image-edit'/label/'result.json').read_text())
 if label=='baseline-bcg-probe' and result.returncode:
  assert data['error'] and data.get('bcg_invalid_signals'),data
 else:assert result.returncode==0 and not data['error'],label
print('Native Joy edit admission/profile and actual BCG applicability recorded')
