"""LongCat GELU+cat native validation and saved-request ABBA."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
candidate = root/'candidate-longcat-gelu-cat'
env = os.environ | {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '8',
                    'FLASHINFER_DISABLE_VERSION_CHECK': '1', 'PYTHONPATH': str(candidate/'python')}
assert (root/'longcat-combined-abba.exit').read_text().strip() == '0'

def run(command, label, cwd=candidate):
    with (root/f'{label}.log').open('w') as log:
        p = subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    (root/f'{label}.exit').write_text(str(p.returncode))
    p.check_returncode()

for mode in ('eager','bcg'):
    for cell in ('a1', 'b1', 'b2', 'a2'):
        repo = candidate if cell.startswith('b') else root/'baseline'
        label = f'combined-{mode}-r2-{cell}'
        flags = ['--bcg'] if mode=='bcg' and cell.startswith('b') else []
        run(['python','-u',str(root/'run-longcat-edit-turbo.py'),'--repo',str(repo),'--label',label,*flags], f'longcat-{label}')
    rows = {cell: json.loads((root/f'artifacts/longcat-image-edit-turbo/combined-{mode}-r2-{cell}/result.json').read_text()) for cell in ('a1','b1','b2','a2')}
    a = (rows['a1']['e2e_latency_s']+rows['a2']['e2e_latency_s'])/2
    b = (rows['b1']['e2e_latency_s']+rows['b2']['e2e_latency_s'])/2
    summary=dict(mode=mode,baseline_mode='original eager',candidate_mode=mode,rows=rows,baseline_mean_s=a,candidate_mean_s=b,reduction_pct=(a-b)/a*100)
    (root/f'artifacts/longcat-image-edit-turbo/combined-{mode}-r2-summary.json').write_text(json.dumps(summary,indent=2))
