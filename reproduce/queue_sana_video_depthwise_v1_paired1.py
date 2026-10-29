"""Wait for the owned A/A admission, then run the pinned balanced comparison once."""
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
receipt = root/'evidence/preflight/sana-video-depthwise-main-eager-aa1-admission.json'
deadline = time.monotonic() + 10800
while not receipt.exists():
    status = json.loads((root/'sana-video-depthwise-main-eager-aa1-status.json').read_text())
    assert status['status'] != 'failed', status
    assert time.monotonic() < deadline, 'A/A admission did not arrive within3hours'
    assert Path('/proc/2355937').exists(), 'Owned A/A wrapper exited without admission; no paired run'
    time.sleep(5)
assert json.loads(receipt.read_text())['passed'], 'A/A noise gate failed; preserve evidence and do not run candidate timings'
print('A/A admitted; starting frozen M-C-O-O-C-M plan', flush=True)
subprocess.run(['python3', str(root/'run_sana_video_blocks1.py'), str(root/'sana-video-depthwise-v1-paired1.json')], check=True)
subprocess.run(['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=', 'bbuf-diffusion-rtx6000-20261026-sgl', 'python3', '/task/admit_sana_video_depthwise_v1_paired1.py'], check=True)
