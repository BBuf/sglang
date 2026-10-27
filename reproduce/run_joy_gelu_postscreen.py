"""Run separate matched profiles, then a focused shape diagnostic after ABBA."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
analysis = json.loads((ROOT / 'joy-gelu-v1-abba1-analysis.json').read_text())
assert analysis['all_planned_blocks_present'] and analysis['source_and_ownership_audit_pass']
assert len(analysis['blocks']) == 4
stages = [
    ('profiles', ['python3', 'run_joy_gelu_profiles.py', '--label', 'gelu-v1-profile1'],
     ROOT / 'joy-gelu-v1-profile1-runner.log'),
    ('shape_probe', ['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'OMP_NUM_THREADS=8',
                     'bbuf-diffusion-rtx6000-20261026-sgl', 'python', '/task/probe_joy_gelu_shapes.py'],
     ROOT / 'joy-gelu-sm120-shape-probe.log'),
]
record = dict(status='running', primary_performance_evidence=False, completed=[])
target = ROOT / 'joy-gelu-v1-postscreen-status.json'
assert not target.exists()
try:
    for name, command, log in stages:
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        record.update(active_stage=name, command=command)
        target.write_text(json.dumps(record, indent=2)+'\n')
        with log.open('xb') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record['completed'].append(name)
        print(json.dumps(dict(stage=name, status='completed')), flush=True)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    record.update(status='completed', active_stage=None)
except Exception as error:
    record.update(status='failed', error=repr(error))
    raise
finally:
    target.write_text(json.dumps(record, indent=2)+'\n')
