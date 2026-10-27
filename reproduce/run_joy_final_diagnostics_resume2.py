"""Keep completed diagnostics; account for native startup and native log transport."""
import hashlib
import json
import subprocess
from pathlib import Path

import queue_joy_final_diagnostics as queue
import run_joy_blocks as blocks
import run_joy_gelu_profiles as profiles
from inspect_joy_native_diagnostic import inspect, read

ROOT = blocks.ROOT
assert queue.identity(2954974) is None and profiles.idle()
assert read(ROOT/'joy-gelu-final-primary1-status.json')['status'] == 'raw_primary_complete_audit_and_final_diagnostics_pending'
previous = read(ROOT/'joy-gelu-final-diagnostics-resume1-status.json')
assert previous['status'] == 'failed' and previous['active'] == 'joy-gelu-final-square-omni-denoise1'
assert previous['error'] == 'AssertionError()'
queue.verify_files()
manifest = read(ROOT/'joy-final-diagnostics-resume2-manifest.json')
for item in manifest['files']:
    assert hashlib.sha256((ROOT/item['remote_relative']).read_bytes()).hexdigest() == item['sha256']
for shape in ('square', 'wide'):
    profile = read(ROOT/'outputs/profiles'/f'joy-gelu-final-{shape}-matched-profile-analysis.json')
    assert profile['standalone_gelu_calls_removed'] == 80
    assert profile['candidate']['activation_calls'] == profile['candidate']['fused_custom_op_calls'] == 80
status_path = ROOT/'joy-gelu-final-diagnostics-resume2-status.json'
status = dict(status='running', candidate_head_sha='4c135ea92d6b4cbb0ead69bb39976f7dbe995455',
    resume_of='joy-gelu-final-diagnostics-resume1-status.json', primary_timing=False,
    profiles=previous['profiles'], omni_denoise=[],
    change='Preserve the one2stepnative startup record separately; correlate every30steprequest and use native synchronizedprofiler log durations omitted by HTTP. No model/observer/primary change or repeated square inference.')
with status_path.open('x') as stream:
    stream.write(json.dumps(status, indent=2)+'\n')


def write_shared(path, data):
    target = str(Path('/task') / path.relative_to(ROOT))
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open('xb').write(sys.stdin.buffer.read())"
    subprocess.run(['docker','exec','-i','bbuf-diffusion-rtx6000-20261026-omni',
                    'python','-c',code,target], input=data, check=True)


try:
    for shape in ('square', 'wide'):
        plan = read(ROOT/f'joy-gelu-final-{shape}-omni-denoise1-plan.json')
        status['active'] = plan['id']
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        folder = ROOT/'outputs/benchmarks'/f'{plan["id"]}-00-omni_compile_profile'
        if shape == 'wide':
            assert not folder.exists()
            blocks.run_block(plan,0,'omni_compile_profile')
        else:
            server = read(ROOT/'block-runs'/folder.name/'metadata.json')
            assert server['plan'] == plan
            assert len(read(folder/'results.json')) == plan['warmups']+plan['repeats']
            assert read(folder/'quality.json')['all_pass']
        correlation, raw_precision = inspect(ROOT,shape)
        write_shared(folder/'runtime-precision.jsonl',raw_precision)
        write_shared(folder/'native-log-correlation.json',(json.dumps(correlation,indent=2)+'\n').encode())
        status['omni_denoise'].append({k:v for k,v in correlation.items() if k != 'correlations'})
    status.update(status='raw_diagnostics_complete_final_archive_audit_pending',active=None)
except Exception as error:
    status.update(status='failed',error=repr(error))
    raise
finally:
    status_path.write_text(json.dumps(status,indent=2)+'\n')
