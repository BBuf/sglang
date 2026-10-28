"""Launch one pinned, task-owned controller only when the GPU pair is idle."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--requires', action='append', default=[])
p.add_argument('--runner', default='run_qwen21_blocks_v2.py')
args = p.parse_args()
root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
assert Path(args.run).name == args.run and Path(args.runner).name == args.runner
assert not (root/(args.run+'-launch.json')).exists()
assert not (root/(args.run+'-status.json')).exists()
for run in args.requires:
    state = json.loads((root/(run+'-status.json')).read_text())
    assert state['status'] == 'raw_complete_audit_pending', state
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
command = ['python3', str(root/args.runner), str(root/(args.run+'.json'))]
with (root/(args.run+'-runner.log')).open('xb') as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
metadata = dict(pid=process.pid, command=command, started=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan_sha256=hashlib.sha256((root/(args.run+'.json')).read_bytes()).hexdigest(), runner_sha256=hashlib.sha256((root/args.runner).read_bytes()).hexdigest(), launch_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/(args.run+'-launch.json')).write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata))
