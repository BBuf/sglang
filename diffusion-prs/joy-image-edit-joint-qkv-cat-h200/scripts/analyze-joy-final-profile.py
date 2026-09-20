"""Run the reusable three-table analyzer on the final Joy model forward."""
from pathlib import Path
import subprocess
import time

root = Path('/campaign')
done = root / 'validate-joy-qkv-cat-final.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
folder = root / 'artifacts/joy-image-edit/qkv-final-candidate-profile'
with (folder / 'triage.txt').open('w') as out:
    subprocess.run(['python', str(root / 'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
        '--framework', 'sglang', '--input', str(folder / 'traces/forward3.trace.json.gz')], stdout=out, check=True)
