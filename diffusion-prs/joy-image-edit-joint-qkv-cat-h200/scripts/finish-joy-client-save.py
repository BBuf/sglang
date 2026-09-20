"""Close only a fully qualifying, visually reviewed Joy model cycle."""
import json
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root / 'audit-joy-client-save.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
art = root / 'artifacts/joy-image-edit'
e = json.loads((art / 'final-evidence.json').read_text())
assert e['qualified'] and e['headline_protocol'] == 'client-save-v1'
assert e['candidate'] == '4811f4d52aa2586412f699b3bb84ed184d76250a'
assert len(e['outputs']) == 141
for group in e['comparisons']['client-save-v1'].values():
    assert len(group['rows']) == 20 and len(group['saved_warmups']) == 4
    assert all(group['means'][k]['reduction_pct'] >= 1.5 for k in ['worker_s', 'client_saved_s', 'client_wall_s'])
assert json.loads((art / 'media-visually-reviewed.json').read_text())['reviewed']
with (root / 'close-joy-image-edit-cycle.log').open('x') as log:
    result = subprocess.run(['python', '-u', str(root / 'close-joy-image-edit-cycle.py')], stdout=log, stderr=subprocess.STDOUT)
(root / 'close-joy-image-edit-cycle.exit').write_text(str(result.returncode))
assert result.returncode == 0
print('Joy final audit qualified; owned cache removed and audited zero', flush=True)
