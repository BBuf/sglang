"""Complete the original three-round contract after the isolated LTX A/A queue."""
import datetime
import json
import subprocess
import time
from pathlib import Path

root=Path('/data/xyzhang/diffusion-6000-20261113')
path=root/'evidence/preflight/nomas-repeat33-controller.json'
assert not path.exists()
state=dict(status='waiting_ltx31',completed=[],cross_framework_performance_admitted=False)

def save(**values):
    state.update(values,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state,indent=2)+'\n')

try:
    save();deadline=time.monotonic()+5*3600
    while True:
        prior=json.loads((root/'evidence/preflight/ltx23-probe-noise31-controller.json').read_text())
        assert not prior['status'].startswith('failed'),prior
        if prior['status']=='complete':break
        assert time.monotonic()<deadline,'prior queue wait deadline'
        time.sleep(15)
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
