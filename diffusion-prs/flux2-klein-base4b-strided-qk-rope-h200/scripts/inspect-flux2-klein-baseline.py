"""Inspect the pinned native Klein baseline and its complete forward."""
import json
from pathlib import Path
import re
import subprocess
import time

root=Path('/campaign');art=root/'artifacts/flux2-klein-base-4b'
done=root/'flux2-klein-base-4b-baseline-profile.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
baseline=json.loads((art/'baseline-a1/result.json').read_text())
profile=json.loads((art/'baseline-profile/result.json').read_text())
assert not baseline['error'] and not profile['error']
assert baseline['output_sha256']==profile['output_sha256']
assert len(profile['native_profile_traces'])==1
trace=Path(profile['native_profile_traces'][0]['path'])
with (art/'baseline-profile/inspect.txt').open('w') as out:
 subprocess.run(['python',str(root/'inspect-flux2-klein-trace.py'),str(trace)],stdout=out,check=True)
with (art/'baseline-profile/triage.txt').open('w') as out:
 subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
  '--framework','sglang','--input',str(trace.parent/'forward3.trace.json.gz')],stdout=out,check=True)
report=json.loads((trace.parent/'forward3-evidence.json').read_text())
log=re.sub(r'\x1b\[[0-9;]*m','',(root/'flux2-klein-base-4b-baseline-a1.log').read_text())
clients=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
assert len(clients)==1
evidence=dict(baseline=baseline,client_saved_s=float(clients[0]),profile_byte_exact=True,profile_forward=report)
(art/'baseline-evidence.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(dict(worker_s=baseline['e2e_latency_s'],denoise_s=baseline['denoise_latency_s'],
 client_saved_s=float(clients[0]),peak_gib=baseline['peak_memory_gb'],model_calls=report['model_calls'],
 kernels=report['kernel_count'],top_kernels=report['top_kernels'][:20]),indent=2))
