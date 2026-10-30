"""Wait for one verified owned runner, then archive its complete raw run once."""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', choices=['joy-input-lifetime-repair1-'+shape+'-'+phase+'1' for shape in ['square','wide'] for phase in ['aa','primary']])
parser.add_argument('--parent-pid', type=int, required=True)
args = parser.parse_args()
root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
status_path = root/(args.run+'-archive-watch-status.json')
assert not status_path.exists(), 'Inspect any previous watcher; never duplicate it'
launch = json.loads((root/(args.run+'-launch.json')).read_text())
assert launch['pid'] == args.parent_pid
expected = 'run_joy_current_head_blocks1.py '+str(root/(args.run+'.json'))


def parent_command():
    result = subprocess.run(['ps', '-p', str(args.parent_pid), '-o', 'args='], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ''


command = parent_command()
assert expected in command, 'An actually live owned runner is required before attaching'
state = dict(status='waiting_for_owned_parent', parent_pid=args.parent_pid, parent_command=command,
             at=datetime.datetime.now(datetime.timezone.utc).isoformat(), no_GPU_work_started=True)
status_path.write_text(json.dumps(state, indent=2)+'\n')
try:
    while command:
        assert expected in command, 'PID was reused; stop without touching its process'
        time.sleep(10)
        command = parent_command()
    raw = json.loads((root/(args.run+'-status.json')).read_text())
    plan = json.loads((root/(args.run+'.json')).read_text())
    assert raw['status'] == 'raw_complete_audit_pending' and len(raw['completed']) == len(plan['order'])
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not (root/(args.run+'-archive.tar.gz')).exists()
    state['status'] = 'archiving_complete_raw_run'
    status_path.write_text(json.dumps(state, indent=2)+'\n')
    subprocess.run(['python3', str(root/'archive_joy_input_lifetime_repair1.py'), args.run], check=True)
    state.update(status='archive_complete_local_transfer_and_audit_pending',
                 manifest=str(root/(args.run+'-archive-manifest.json')))
except Exception as exc:
    state.update(status='stopped_preserve_outputs_no_automatic_retry', error=repr(exc))
    raise
finally:
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status_path.write_text(json.dumps(state, indent=2)+'\n')
