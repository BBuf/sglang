"""Candidate quality/BCG applicability, retaining rejected graph evidence."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root / 'longcat-combined-abba-r2.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
env = os.environ | {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '8',
                    'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING': '1'}
rows = []
for quality in ('lossless', 'high'):
    for cell in ('a1', 'b1', 'b2', 'a2'):
        label = f'quality-{quality}-{cell}'
        command = ['python', '-u', str(root/'run-longcat-quality.py'),
                   '--repo', str(root/'candidate-longcat-gelu-cat'),
                   '--label', label, '--quality', quality]
        if cell.startswith('b'):
            command.append('--bcg')
        with (root/f'longcat-{label}.log').open('w') as log:
            p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        (root/f'longcat-{label}.exit').write_text(str(p.returncode))
        result = root/f'artifacts/longcat-image-edit-turbo/{label}/result.json'
        assert result.exists(), label
        row = json.loads(result.read_text())
        rows.append(row)
        # Failed graph applicability is evidence, never a timing comparator.
        if p.returncode:
            assert cell.startswith('b') and row.get('error'), row
        else:
            assert not row.get('error'), row
(root/'artifacts/longcat-image-edit-turbo/quality-matrix.json').write_text(
    json.dumps(rows, indent=2))
