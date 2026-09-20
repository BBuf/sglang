"""Read final paired native profiler evidence after generation has completed."""
import json
from pathlib import Path
import subprocess
import time
root=Path('/campaign'); done=root/'measure-klein-qk-v3.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0'
for label in ['baseline-profile','qk-v3-candidate-profile']:
 art=root/'artifacts/flux2-klein-base-4b'/label
 row=json.loads((art/'result.json').read_text());trace=Path(row['native_profile_traces'][0]['path'])
 with (art/'qk-chain-inspect.txt').open('w') as out:
  subprocess.run(['python',str(root/'inspect-klein-qk-chain.py'),str(trace)],stdout=out,check=True)
 if label=='qk-v3-candidate-profile':
  with (art/'triage.txt').open('w') as out:
   subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),'--framework','sglang','--input',str(trace.parent/'forward3.trace.json.gz')],stdout=out,check=True)
 print(label,(trace.parent/'qk-chain-evidence.json').read_text()[:130],flush=True)
