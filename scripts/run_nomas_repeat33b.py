"""Recover independent Nomas repeats after LTX own-main output mismatch; original failed controllers retained."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/nomas-repeat33b-controller.json'
assert not path.exists()
state=dict(status='waiting_mismatch38_CPU_audit',completed=[],cross_framework_performance_admitted=False)

def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')

try:
    save();deadline=time.monotonic()+3600
    assert json.loads((root/'evidence/preflight/nomas-long-scope29-controller.json').read_text())['status']=='complete'
    assert json.loads((root/'evidence/preflight/nomas-repeat33-controller.json').read_text())['completed']==[]
    while True:
        audit=root/'evidence/quality/ltx23-noise-mismatch38/summary.json'
        if audit.exists() and json.loads(audit.read_text()).get('status')=='diagnostic_complete_no_performance_admission':break
        assert time.monotonic()<deadline,'CPU diagnostic deadline'
        time.sleep(5)
    for v in json.loads((root/'inputs/nomas-repeat33-queue.json').read_text())['visits']:
        assert not (root/'runs'/v['run']).exists(), 'Do not overwrite or selectively reuse started visits'
    queue=json.loads((root/'inputs/nomas-repeat33-queue.json').read_text())['visits']
    assert len(queue)==10
    for v in queue:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        command=['python3',str(root/v['script']),v['arm'],v['run'],v['plan']]
        save(status='quiet_running',active=v['run'],command=command)
        subprocess.run(command,check=True)
        folder=root/'runs'/v['run'];j=json.loads((folder/'status.json').read_text())
        assert j['status']=='raw_complete' and j['gpu_cleanup']=='idle_verified'
        rows=json.loads((folder/'outputs/results.json').read_text())
        assert len(rows)==8 and all(r['matches_reference'] and not r['profiler'] for r in rows)
        state['completed'].append(v['run']);save()
    save(status='complete',active=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry',error=repr(exc));raise
