"""Separate final-head model traces and native Omni denoise diagnostics; no primary timing."""
import hashlib
import json
import statistics
import subprocess

import queue_joy_final_diagnostics as queue
from pathlib import Path

import analyze_joy_gelu_profiles as profile_analysis
import run_joy_blocks as blocks
import run_joy_gelu_profiles as profiles

ROOT = blocks.ROOT
primary = json.loads((ROOT/'joy-gelu-final-primary1-status.json').read_text())
assert primary['status']=='raw_primary_complete_audit_and_final_diagnostics_pending'
head = primary['protocol']['candidate_head_sha']
previous = json.loads((ROOT/'joy-gelu-final-diagnostics1-status.json').read_text())
assert previous['status']=='failed' and previous['error']=="PermissionError(13, 'Permission denied')"
assert queue.identity(2857472) is None and queue.identity(2932533) is None
assert profiles.idle(), 'Both GPUs must be idle before the controlled resume'
queue.verify_files()
status_path = ROOT/'joy-gelu-final-diagnostics-resume1-status.json'
assert not status_path.exists()
status = dict(status='running', candidate_head_sha=head, profiles=[], omni_denoise=[], primary_timing=False,
              resume_of='joy-gelu-final-diagnostics1-status.json',
              change='Write task-owned container-created outputs through their container; reuse the two complete square traces without new model execution.')


def read(path):
    return json.loads(path.read_text())


def write_shared(path, data, arm):
    # The output directories were created by the task container's root user.
    # Write only a new task artifact through that same container; no chmod/chown.
    target = str(Path('/task') / path.relative_to(ROOT))
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.open('xb').write(sys.stdin.buffer.read())"
    subprocess.run(['docker','exec','-i','bbuf-diffusion-rtx6000-20261026-'+arm,
                    'python','-c',code,target],input=data,check=True)


try:
    for shape in ('square','wide'):
        contract=Path('/task/inputs/joy-workloads'+('' if shape=='square' else '-wide')+'.json')
        label=f'gelu-final-{shape}-profile1'
        analyses={}
        for source in ('main','pr'):
            status.update(active=f'{label}-{source}')
            status_path.write_text(json.dumps(status,indent=2)+'\n')
            if shape != 'square':
                profiles.run(source,label,['high'],head,hashlib.sha256(b'').hexdigest(),contract)
            else:
                complete = read(ROOT/'block-runs'/f'joy-{label}-{source}'/'metadata.json')
                assert complete['status']=='profile_generated_not_primary_timing'
                assert complete['source_sha']==(head if source=='pr' else 'af21e2b989e482cfdce39689126562f2209d53ac')
                assert complete['diff_sha256']==hashlib.sha256(b'').hexdigest()
                assert complete['contract_sha256']==hashlib.sha256((ROOT/contract.relative_to('/task')).read_bytes()).hexdigest()
                assert complete['runner_sha256']==hashlib.sha256((ROOT/'run_joy_gelu_profiles.py').read_bytes()).hexdigest()
            name=f'joy-{label}-{source}'
            rows=read(ROOT/'outputs/qualification'/f'sgl-joy-{label}-{source}-high'/'results.json')
            reference=('sgl-joy-gelu-v1-main-high' if shape=='square' else 'sgl-joy-gelu-v4-wide-main-high') if source=='main' else f'sgl-joy-gelu-v4-{shape}-pr-high'
            assert len(rows)==1 and rows[0]['sha256']==read(ROOT/'outputs/qualification'/reference/'results.json')[0]['sha256']
            metadata=read(ROOT/'block-runs'/name/'metadata.json')
            assert len(metadata['traces'])==1
            trace=Path(metadata['traces'][0]['path'])
            assert hashlib.sha256(trace.read_bytes()).hexdigest()==metadata['traces'][0]['sha256']
            analyses[source]=profile_analysis.analyze(trace,3)
        assert analyses['main']['gelu_cpu_calls']-analyses['pr']['gelu_cpu_calls']==80
        assert analyses['pr']['fused_custom_op_calls']==analyses['pr']['activation_calls']==80
        target=ROOT/'outputs/profiles'/f'joy-gelu-final-{shape}-matched-profile-analysis.json'
        write_shared(target,(json.dumps(dict(shape=shape,main=analyses['main'],candidate=analyses['pr'],
            standalone_gelu_calls_removed=80,primary_timing=False),indent=2)+'\n').encode(),'sgl')
        status['profiles'].append(str(target))
    for shape in ('square','wide'):
        plan=read(ROOT/f'joy-gelu-final-{shape}-omni-denoise1-plan.json')
        status.update(active=plan['id'])
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        precision_dir=ROOT/'outputs/precision/omni-native'
        before_precision={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in precision_dir.glob('*.jsonl')}
        blocks.run_block(plan,0,'omni_compile_profile')
        folder=ROOT/'outputs/benchmarks'/f'{plan["id"]}-00-omni_compile_profile'
        precision_files=[p for p in precision_dir.glob('*.jsonl')
                         if before_precision.get(p.name)!=hashlib.sha256(p.read_bytes()).hexdigest()]
        assert len(precision_files)==1, 'Missing or ambiguous actual worker precision metadata'
        precision_bytes=precision_files[0].read_bytes()
        precision_rows=[json.loads(line) for line in precision_bytes.splitlines()]
        assert len(precision_rows)==4+plan['warmups']+plan['repeats']
        assert all(len(row['scheduler_calls'])==30 for row in precision_rows)
        write_shared(folder/'runtime-precision.jsonl',precision_bytes,'omni')
        rows=read(folder/'results.json')
        values=[]
        for row in rows:
            response=read(folder/f'{row["index"]:02d}'/'response.json')
            stages=response['metrics']['stage_durations']
            value=stages[plan['required_stage_metric']]
            assert 0<value<row['http_seconds']
            if not row['warmup']:
                values.append(value)
        status['omni_denoise'].append(dict(shape=shape,seconds=values,p50_seconds=statistics.median(values),
            metric=plan['required_stage_metric'],primary_timing=False,
            precision_metadata=str(folder/'runtime-precision.jsonl'),
            observer_note='Metadata-only scheduler wrappers inside separate diagnostic runs; no extra tensor reads/copies/synchronizations, no primary E2E use.'))
    status.update(status='raw_diagnostics_complete_final_archive_audit_pending',active=None)
except Exception as error:
    status.update(status='failed',error=repr(error))
    raise
finally:
    status_path.write_text(json.dumps(status,indent=2)+'\n')
