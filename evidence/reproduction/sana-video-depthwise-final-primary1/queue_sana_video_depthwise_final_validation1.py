"""Wait for the frozen paired admission, prepare a clean final tree, then validate once."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
container = 'bbuf-diffusion-rtx6000-20261026-sgl'
status_path = root/'sana-video-depthwise-final-validation1-queue-status.json'
plan_path = root/'sana-video-depthwise-final-validation1-queue-plan.json'
plan = json.loads(plan_path.read_text())
receipt = root/'evidence/preflight/sana-video-depthwise-v1-paired1-admission.json'

def status(name, **kwargs):
    status_path.write_text(json.dumps(dict(status=name, **kwargs), indent=2)+'\n')

assert not status_path.exists(), 'No automatic retry'
try:
    status('waiting_for_V1_primary_admission')
    deadline = time.monotonic()+21600
    while not receipt.exists():
        p = root/'sana-video-depthwise-v1-paired1-status.json'
        if p.exists():
            assert not json.loads(p.read_text())['status'].startswith('failed'), p
        assert time.monotonic()<deadline
        assert Path('/proc/2373143').exists(), 'Paired queue exited without admission'
        time.sleep(5)
    assert json.loads(receipt.read_text())['passed'], 'V1 primary failed; preserve everything'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for path, expected in plan['sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected, path
    status('preparing_final_worktree')
    assert not (root/'candidates/sana-video-depthwise-glu-final').exists()
    subprocess.run(['docker','exec',container,'git','-C','/task/repos/sglang','fetch','/task/evidence/preflight/sana-video-depthwise-final-head1.bundle',plan['head']],check=True)
    subprocess.run(['docker','exec',container,'git','-C','/task/repos/sglang','worktree','add','--detach','/task/candidates/sana-video-depthwise-glu-final',plan['head']],check=True)
    assert subprocess.check_output(['docker','exec',container,'git','-C','/task/candidates/sana-video-depthwise-glu-final','rev-parse','HEAD'],text=True).strip()==plan['head']
    assert not subprocess.check_output(['docker','exec',container,'git','-C','/task/candidates/sana-video-depthwise-glu-final','status','--porcelain'])
    status('running_final_tests', head=plan['head'])
    subprocess.run(['python3',str(root/'run_sana_video_depthwise_final_tests1.py'),'--head',plan['head']],check=True)
    status('running_native_quality_and_BCG_diagnostics', head=plan['head'])
    subprocess.run(['python3',str(root/'run_sana_video_depthwise_final_compatibility1.py')],check=True)
    subprocess.run(['docker','exec','-e','CUDA_VISIBLE_DEVICES=',container,'python3','/task/audit_sana_video_depthwise_final_compatibility1.py'],check=True)
    status('raw_compatibility_passed_actual_profile_and_visual_audit_pending', head=plan['head'])
except Exception as exc:
    status('failed_preserve_all_outputs_no_retry', error=repr(exc))
    raise
