"""Run the pinned read-only audit once, after successful native qualification."""
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status = root / 'sana-video-native-qualification1-status.json'
while True:
    value = json.loads(status.read_text())
    if value['status'] == 'raw_complete_video_quality_runtime_inputs_and_profile_audit_pending':
        break
    if value['status'].startswith('failed'):
        raise RuntimeError(value)
    time.sleep(5)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert (root/'sana-video-observer1/sitecustomize.py').read_bytes() == (root/'sana_video_tensor_sitecustomize.py').read_bytes()
subprocess.run(['python3', str(root/'run_sana_video_tensor_audit1.py')], check=True)
