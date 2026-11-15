"""Two sequential unprofiled main noise visits; stop on any output or isolation failure."""
import datetime
import json
import subprocess
from pathlib import Path

root = Path('/data/xyzhang/diffusion-6000-20261113')
path = root / 'evidence/preflight/nomas-noise10-controller.json'
assert not path.exists()
quality = root / 'runs/nomas-49-sgl-mean-quality07'
admission = json.loads((quality / 'status.json').read_text())
assert admission['status'] == 'raw_complete' and admission['gpu_cleanup'] == 'idle_verified'
assert len(json.loads((quality / 'outputs/results.json').read_text())) == 7
state = dict(status='running', completed=[], comparison_scope='within_sglang_main_AA', cross_framework_performance_admitted=False)


def save(**values):
    state.update(values, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state, indent=2)+'\n')


try:
    for suffix in ('A1', 'A2'):
        name = 'nomas-49-sgl-noise10-' + suffix
        command = ['python3', str(root/'scripts/nomas-quiet-visits/run.py'), 'sgl', name, 'inputs/'+name+'.json']
        save(active_run=name, command=command)
        subprocess.run(command, check=True)
        status = json.loads((root/'runs'/name/'status.json').read_text())
        assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified'
        state['completed'].append(name)
        save()
    save(status='complete', active_run=None)
except BaseException as exc:
    save(status='failed_preserve_no_automatic_retry', error=repr(exc))
    raise
