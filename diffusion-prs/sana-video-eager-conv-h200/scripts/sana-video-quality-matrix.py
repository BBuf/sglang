"""Record lossless/high eager and real-BCG applicability after native A/B."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
repo = root/'candidate-sana-video'
done = root/'validate-sana-video-conv.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
rows = []
for mode in ('eager','bcg'):
    for arm in ('a1','b1','b2','a2'):
        quality = 'lossless' if arm.startswith('a') else 'high'
        label = f'quality-{mode}-{arm}'
        log = root/f'sana-video-{label}.log'
        assert not log.exists()
        args = ['python','-u',str(root/'run-sana-video.py'),
            '--repo',str(repo),'--label',label,'--quality',quality]
        if mode == 'bcg':
            args.append('--bcg')
        with log.open('w') as stream:
            p = subprocess.run(args,cwd=repo,stdout=stream,stderr=subprocess.STDOUT)
        (root/f'sana-video-{label}.exit').write_text(str(p.returncode))
        result = json.loads((root/'artifacts/sana-video'/label/'result.json').read_text())
        if p.returncode:
            assert mode == 'bcg' and result.get('bcg_invalid_signals'), result
        rows.append(result)
(root/'artifacts/sana-video/quality-matrix.json').write_text(json.dumps(rows,indent=2))
print('Completed',len(rows),'quality/applicability cells')
