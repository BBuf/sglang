"""Inspect the pinned native Joy image edit baseline and its complete forward."""
import json
from pathlib import Path
import re
import subprocess
import time

root=Path('/campaign');art=root/'artifacts/joy-image-edit'
done=root/'joy-image-edit-baseline-profile-retry1.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
baseline=json.loads((art/'baseline-retry1/result.json').read_text())
profile=json.loads((art/'baseline-profile-retry1/result.json').read_text())
assert not baseline['error'] and not profile['error']
assert baseline['output_sha256']==profile['output_sha256']
rank0=[p for p in profile['native_profile_traces'] if 'global-rank0' in p['path']]
assert len(rank0)==1,profile['native_profile_traces']
trace=Path(rank0[0]['path'])
with (art/'baseline-profile-retry1/inspect.txt').open('w') as out:
 subprocess.run(['python',str(root/'inspect-joy-image-edit-trace.py'),str(trace)],stdout=out,check=True)
with (art/'baseline-profile-retry1/triage.txt').open('w') as out:
 subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
  '--framework','sglang','--input',str(trace.parent/'forward3.trace.json.gz')],stdout=out,check=True)
report=json.loads((trace.parent/'forward3-evidence.json').read_text())
log=re.sub(r'\x1b\[[0-9;]*m','',(root/'joy-image-edit-baseline-retry1.log').read_text())
clients=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
assert len(clients)==1
from PIL import Image
image=Image.open(next((art/'baseline-retry1').glob('*.png')))
evidence=dict(actual_image_size=list(image.size),baseline=baseline,client_saved_s=float(clients[0]),profile_byte_exact=True,profile_forward=report)
(art/'baseline-evidence.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(dict(actual_image_size=list(image.size),worker_s=baseline['e2e_latency_s'],denoise_s=baseline['denoise_latency_s'],
 client_saved_s=float(clients[0]),peak_gib=baseline['peak_memory_gb'],model_calls=report['model_calls'],
 kernels=report['kernel_count'],top_kernels=[row|{'name':row['name'][:160]} for row in report['top_kernels'][:20]]),indent=2))
