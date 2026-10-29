"""After owned compatibility terminates, audit raw evidence and archive it once."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status_path = root/'sana-video-ln-final-archive1-queue-status.json'
assert not status_path.exists()
manifest = json.loads((root/'sana-video-ln-final-archive1-manifest.json').read_text())


def save(status, **fields):
    status_path.write_text(json.dumps(dict(status=status, **fields), indent=2)+'\n')


try:
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    save('waiting_for_final_compatibility_raw_completion', owned_runner_pid=1716826)
    deadline = time.monotonic()+10800
    while True:
        path = root/'sana-video-ln-final-compatibility1-status.json'
        state = json.loads(path.read_text())
        assert not state['status'].startswith('failed'), state
        if state['status'] == 'raw_complete_quality_replay_and_profile_audit_pending':
            break
        assert time.monotonic() < deadline, 'Compatibility deadline exceeded'
        assert Path('/proc/1716826').exists(), 'Owned compatibility runner exited before raw completion'
        time.sleep(5)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    save('auditing_raw_videos_and_replay')
    subprocess.run(['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=', 'bbuf-diffusion-rtx6000-20261026-sgl',
                    'python3', '/task/audit_sana_video_final_compatibility1.py'], check=True)
    save('archiving_completed_qualification')
    subprocess.run(['python3', str(root/'archive_sana_video_ln_qualification1.py')], check=True)
    save('archive_complete_local_verification_profile_and_visual_review_pending',
         archive='sana-video-ln-qualification1-archive.tar.gz',
         final_primary_started=False, new_pr_created=False)
except Exception as exc:
    save('failed_preserve_outputs_no_retry', error=repr(exc))
    raise
