"""Archive after owned final validation; never overlap a primary timing run."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status_path = root/'sana-video-depthwise-archive2-queue-status.json'
assert not status_path.exists()
plan = json.loads((root/'sana-video-depthwise-archive2-queue-plan.json').read_text())

def status(value, **kwargs):
    status_path.write_text(json.dumps(dict(status=value,**kwargs),indent=2)+'\n')

try:
    for path, sha in plan['sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha,path
    status('waiting_for_final_validation_raw_audit',owned_runner_pid=2405453)
    deadline = time.monotonic()+21600
    while True:
        current=json.loads((root/'sana-video-depthwise-final-validation1-queue-status.json').read_text())
        assert not current['status'].startswith('failed'),current
        if current['status']=='raw_compatibility_passed_actual_profile_and_visual_audit_pending':break
        assert time.monotonic()<deadline
        assert Path('/proc/2405453').exists(),'Final validation exited without successful raw audit'
        time.sleep(5)
    status('archiving')
    subprocess.run(['python3',str(root/'archive_sana_video_depthwise_qualification2.py')],check=True)
    status('archive_complete_local_verification_profile_and_visual_review_pending',archive='sana-video-depthwise-qualification1-archive.tar.gz',final_primary_started=False,new_pr_created=False)
except Exception as exc:
    status('failed_preserve_outputs_no_retry',error=repr(exc))
    raise
