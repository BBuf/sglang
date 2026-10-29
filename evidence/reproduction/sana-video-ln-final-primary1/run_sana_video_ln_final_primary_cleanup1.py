"""Run the frozen final paired model comparison only after complete admission."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
plan_path = root/'sana-video-ln-final-primary-cleanup1.json'
assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == '53cae0f00bccd5fbb3e00d367f26e793acea0226a6ff7c4ad0a087affa94c285'
plan = json.loads(plan_path.read_text())
assert not (root/(plan['id']+'-status.json')).exists(), 'No automatic retry'
for name in plan['required_admissions']:
    assert json.loads((root/name).read_text())['passed'], name
compatibility = json.loads((root/'evidence/preflight/sana-video-ln-final-compatibility1-admission.json').read_text())
assert compatibility['final_primary_admitted'] and compatibility['source_sha'] == 'e0d694593a65cb5b463c4011b84ee50abefa5e43'
archive = json.loads((root/'sana-video-ln-final-archive1-queue-status.json').read_text())
assert archive['status'] == 'archive_complete_local_verification_profile_and_visual_review_pending'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
subprocess.run(['python3', str(root/'run_sana_video_blocks1.py'), str(plan_path)], check=True)
subprocess.run(['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=', 'bbuf-diffusion-rtx6000-20261026-sgl',
                'python3', '/task/admit_sana_video_ln_final_primary_cleanup1.py'], check=True)
subprocess.run(['python3', str(root/'archive_sana_video_ln_final_primary1.py')], check=True)
