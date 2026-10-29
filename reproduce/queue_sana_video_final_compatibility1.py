"""After admitted V1 timing, materialize the frozen final head, test, and diagnose."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status_path = root/'sana-video-ln-final-compatibility1-queue-status.json'
assert not status_path.exists(), 'No automatic retry'
manifest = json.loads((root/'sana-video-ln-final-queue-manifest1.json').read_text())

def save(status, **fields):
    status_path.write_text(json.dumps(dict(status=status, **fields), indent=2)+'\n')

try:
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
    save('waiting_for_v1_primary_admission')
    receipt = root/'evidence/preflight/sana-video-ln-v1-paired1-admission.json'
    deadline = time.monotonic()+14400
    while not receipt.exists():
        assert time.monotonic()<deadline, 'V1 admission timeout'
        if not Path('/proc/1513445').exists():
            assert receipt.exists(), 'V1 queue exited without an admission receipt'
            break
        time.sleep(5)
    assert json.loads(receipt.read_text())['passed'], 'V1 performance gate failed; do not run final diagnostics'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    head = manifest['head']
    container = 'bbuf-diffusion-rtx6000-20261026-sgl'
    destination = root/'candidates/sana-video-eager-ln-final'
    assert not destination.exists(), 'Do not overwrite an existing remote worktree'
    save('materializing_final_source', head=head)
    subprocess.run(['docker', 'exec', container, 'git', '-C', '/task/repos/sglang', 'fetch', '/task/sana-video-ln-final-head1.bundle', 'HEAD'], check=True)
    subprocess.run(['docker', 'exec', container, 'git', '-C', '/task/repos/sglang', 'worktree', 'add', '--detach', '/task/candidates/sana-video-eager-ln-final', head], check=True)
    observer = root/'sana-video-bcg-observer1'
    observer.mkdir(exist_ok=False)
    shutil.copyfile(root/'sana_video_bcg_sitecustomize1.py', observer/'sitecustomize.py')
    save('running_final_tests', head=head)
    subprocess.run(['python3', str(root/'run_sana_video_ln_final_tests1.py'), '--head', head], check=True)
    save('running_final_compatibility', head=head)
    subprocess.run(['python3', str(root/'run_sana_video_final_compatibility1.py')], check=True)
    save('raw_complete_audit_pending', head=head)
except Exception as exc:
    save('failed_preserve_outputs_no_retry', error=repr(exc))
    raise
