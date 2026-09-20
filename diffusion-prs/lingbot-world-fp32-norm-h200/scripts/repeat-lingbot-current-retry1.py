"""Two predeclared additional ABBA groups after a post-save client outlier; retain every earlier row."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
repo = root/'candidate-lingbot-current'
done = root/'finish-lingbot-current-retry1.exit'
while not done.exists(): time.sleep(5)
assert done.read_text().strip() == '0', done
base = json.loads((root/'artifacts/lingbot-world-v2/baseline-a1/result.json').read_text())

def run(label, source, flags=(), disabled=False):
    log = root/f'lingbot-{label}.log'
    with log.open('x') as out:
        result = subprocess.run(['python','-u',str(root/'run-lingbot-world-v2.py'),
            '--repo',str(source),'--label',label,*flags], stdout=out, stderr=subprocess.STDOUT)
    (root/f'lingbot-{label}.exit').write_text(str(result.returncode))
    record = json.loads((root/'artifacts/lingbot-world-v2'/label/'result.json').read_text())
    if result.returncode:
        assert disabled and record['error'] and '[diffusion bcg] disabled' in log.read_text().lower(), label
    else:
        assert not disabled, label+' unexpectedly enabled BCG'
        if record['quality'] == 'lossless':
            assert record['output_sha256'] == base['output_sha256'], label
    return record

for repeat in (3,4):
    for arm in ('a1','b1','b2','a2'):
        run(f'current-r{repeat}-{arm}', root/'baseline-current' if arm.startswith('a') else repo)
print('Both additional fixed native ABBA groups completed')
