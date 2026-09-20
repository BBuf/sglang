"""Admit the next public checkpoint only after the preceding model is closed."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/campaign');done=root/'close-flux2-klein-base-4b-cycle.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
assert not (root/'model-caches/flux2-klein-base-4b-cycle').exists()
assert (root/'prepare-joy-image-edit.exit').read_text().strip()=='0'
for label,flags in [('baseline-retry1',[]),('baseline-profile-retry1',['--profile']),('baseline-bcg-probe-retry1',['--bcg'])]:
 with (root/f'joy-image-edit-{label}.log').open('x') as log:
  result=subprocess.run(['python','-u',str(root/'run-joy-image-edit-retry1.py'),'--label',label,*flags],stdout=log,stderr=subprocess.STDOUT)
 (root/f'joy-image-edit-{label}.exit').write_text(str(result.returncode))
 data=json.loads((root/'artifacts/joy-image-edit'/label/'result.json').read_text())
 if label=='baseline-bcg-probe-retry1' and result.returncode:
  assert data['error'] and data.get('bcg_invalid_signals'),data
 else:assert result.returncode==0 and not data['error'],label
print('Native Joy edit admission/profile and actual BCG applicability recorded')
