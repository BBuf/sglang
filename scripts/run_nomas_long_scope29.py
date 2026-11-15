"""Finish frozen long-video resource coverage after the current owned profile."""
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/data/xyzhang/diffusion-6000-20261113')
path = root / 'evidence/preflight/nomas-long-scope29-controller.json'
assert not path.exists()
state = dict(status='waiting_LTX_profile28', benchmark=False, completed=[])

def save(**updates):
    state.update(updates, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2) + '\n')

try:
    save()
    deadline = time.monotonic() + 3 * 3600
    while True:
        previous = json.loads((root / 'runs/ltx23-full-sgl-profile28/status.json').read_text())
        assert previous['status'] != 'failed_preserve_no_retry', previous
        if previous['status'] == 'raw_complete' and previous.get('gpu_cleanup') == 'idle_verified':
            break
        assert time.monotonic() < deadline, 'prior profile deadline; inspect without retry'
        time.sleep(15)
    queue = json.loads((root / 'inputs/nomas-long-scope29-queue.json').read_text())
    for visit in queue:
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        plan = json.loads((root / visit['plan']).read_text())
        if visit['mode'] == 'c2':
            for name, digest in plan['comparison_reference_sha256'].items():
                assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        command = ['python3', str(root / visit['script']), visit['arm'], visit['run'], visit['plan']]
        save(status='running', active=visit['run'], command=command)
        subprocess.run(command, check=True)
        folder = root / 'runs' / visit['run']
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified', status
        rows = json.loads((folder / 'outputs/results.json').read_text())
        assert len(rows) == len(plan['requests']) and all(x['matches_reference'] for x in rows)
        if visit['mode'] == 'sp2':
            assert rows[0]['sha256'] == rows[2]['sha256']
        else:
            assert all(x['actual_overlapping_jobs'] for x in rows)
            for offset in [0, 6]:
                assert rows[offset]['sha256'] == rows[offset + 4]['sha256']
                assert rows[offset + 1]['sha256'] == rows[offset + 5]['sha256']
        state['completed'].append(visit['run'])
        save()
    save(status='complete', active=None, note='Every output exact to archived same-arm SP1/C1; no new cross-framework quality pass or performance claim.')
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry', error=repr(exc))
    raise
