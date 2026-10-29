"""Gate fresh A/A on exact native artifacts and actual fused model dispatch."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status_file = root/'sana-video-depthwise-profile-aa1-queue-status.json'
assert not status_file.exists()
def save(status, **fields):
    status_file.write_text(json.dumps(dict(status=status,**fields),indent=2)+'\n')
try:
    manifest = json.loads((root/'sana-video-depthwise-profile-aa1-queue-manifest.json').read_text())
    for path,expected in manifest['sha256'].items():
        actual = root/(Path(path).name if path.startswith('scripts/') or path.startswith('evidence/plans/') else path)
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected,path
    save('waiting_native_quality')
    while True:
        state = json.loads((root/'sana-video-depthwise-v1-qualification2-status.json').read_text())
        assert not state['status'].startswith('failed'),state
        if state['status'] == 'raw_complete_MP4_exact_main_profile_audit_pending':
            assert len(state['comparisons']) == 5 and all(x['MP4_exact_main'] for x in state['comparisons'])
            break
        time.sleep(10)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    save('auditing_actual_profile')
    subprocess.run(['docker','exec','bbuf-diffusion-rtx6000-20261026-sgl','python3','/task/analyze_sana_video_depthwise_v1_profile1.py'],check=True)
    profile = json.loads((root/'evidence/preflight/sana-video-depthwise-v1-profile1.json').read_text())
    assert profile['actual_fused_path_verified']
    save('running_fresh_AA',profile_receipt='evidence/preflight/sana-video-depthwise-v1-profile1.json')
    subprocess.run(['python3',str(root/'run_sana_video_depthwise_aa1.py')],check=True)
    admission = json.loads((root/'evidence/preflight/sana-video-depthwise-main-eager-aa1-admission.json').read_text())
    save('complete' if admission['passed'] else 'AA_noise_failed',admission=admission)
except Exception as exc:
    save('failed',error=repr(exc))
    raise
