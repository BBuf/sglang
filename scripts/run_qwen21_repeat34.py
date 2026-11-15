"""Complete Qwen21 final-codec three-round protocol after LTX BCG audit32."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/qwen21-repeat34-controller.json'
assert not path.exists()
state=dict(status='waiting_ltx32',completed=[],benchmark=False)

def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')

try:
    save();deadline=time.monotonic()+7*3600
    while True:
        prior=json.loads((root/'evidence/preflight/ltx23-bcg32-controller.json').read_text())
        if prior['status']=='raw_complete_requires_capture_replay_audit':break
        if prior['status'].startswith('failed'):
            # BCG quality/capture failure does not invalidate an independent
            # Qwen run, but an uncleared process or failed predecessor does.
            clean=root/'runs/ltx23-full-sgl-bcg-audit32/status.json'
            assert clean.exists() and json.loads(clean.read_text()).get('gpu_cleanup')=='idle_verified',prior
            assert json.loads((root/'evidence/preflight/nomas-repeat33-controller.json').read_text())['status']=='complete'
            state['prior_LTX_failure_retained']=prior
            break
        assert time.monotonic()<deadline,'prior queue wait deadline'
        time.sleep(15)
    queue=json.loads((root/'inputs/qwen21-repeat34-queue.json').read_text())
    assert len(queue['visits'])==16
    for v in queue['visits']:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        command=['python3',str(root/v['script']),v['arm'],v['run'],v['plan']]
        save(status='quiet_running',benchmark=True,active=v['run'],command=command)
        subprocess.run(command,check=True)
        folder=root/'runs'/v['run'];j=json.loads((folder/'status.json').read_text())
        assert j['status']=='raw_complete' and j['gpu_cleanup']=='idle_verified'
        assert j['head_sha']==queue['heads'][v['mode']]
        rows=json.loads((folder/'outputs/results.json').read_text())
        assert len(rows)==9 and all(r['matches_reference'] and r['native_steps_observed']==50 for r in rows)
        state['completed'].append(v['run']);save()
    save(status='complete',active=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc));raise
