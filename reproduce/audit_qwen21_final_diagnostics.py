"""Audit separate native stage timings; never derive primary E2E from these runs."""
import argparse
import base64
import hashlib
import json
import re
import statistics
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('size',type=int,choices=[512,1024])
args=p.parse_args()
run=f'qwen21-gate-up-v4-final-{args.size}-diagnostics1'
plan=json.loads((ROOT/'evidence/plans'/f'{run}.json').read_text())
checks={}
for index,config in enumerate(plan['order']):
    spec=plan['configurations'][config]
    arm=spec['arm']
    name=f'{run}-{index:02d}-{config}'
    server=ROOT/'evidence/block-runs'/name
    source=json.loads((server/'metadata.json').read_text())
    assert source['status']=='complete' and source['source_sha']==spec['head']
    assert source['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
    launcher=(server/'launch.sh').read_text()
    assert ('SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1' in launcher) if arm=='sgl' else ('--enable-diffusion-pipeline-profiler' in launcher)
    all_requests=[]
    task_results={}
    for task in spec.get('tasks',plan['tasks']):
        folder=ROOT/'evidence/benchmarks'/f'{name}-{task}'
        metadata=json.loads((folder/'metadata.json').read_text())
        assert metadata['profiler'] and metadata['input_sha256']==plan['contract_sha256'][task]
        q=ROOT/'evidence/qualification'/f'{arm}-qwen21-base-{task}-{name}-high'
        qualification=json.loads((q/'results.json').read_text())
        assert len(qualification)==4
        qualified=qualification[0]['sha256']
        all_requests += [(task,q/Path(row['request']).name,'qualification',row) for row in qualification]
        actual=json.loads((folder/'results.json').read_text())
        assert len(actual)==plan['warmups']+plan['repeats']
        allowed=set(metadata['args']['allowed_gpu_pids'])
        assert allowed==set(source['gpu_container_pids'][0])
        denoise=[]
        if arm=='sgl':
            ref=ROOT/'evidence/qualification'/plan['main_qualification_references'][task]
            assert qualified==json.loads((ref/'results.json').read_text())[0]['sha256']
        for i,row in enumerate(actual):
            directory=folder/f'{i:02d}'
            request=json.loads((directory/'request.json').read_text())
            response=json.loads((directory/'response.json').read_text())
            png=(directory/'output.png').read_bytes()
            assert hashlib.sha256(png).hexdigest()==row['sha256']==qualified
            assert base64.b64decode(response['data'][0]['b64_json'],validate=True)==png
            with Image.open(directory/'output.png') as im:
                assert im.mode=='RGBA' and im.size==(args.size,args.size)
            assert row['warmup']==(i<plan['warmups'])
            assert int(request['num_inference_steps'])==50 and int(request['seed'])==1101
            assert int(request['output_compression'])==(0 if arm=='sgl' else 100)
            for sample in json.loads((directory/'gpu-samples.json').read_text()):
                assert set(sample['gpus'][0]['pids'])<=allowed and not sample['gpus'][1]['pids']
            phase='warmup' if row['warmup'] else 'diagnostic'
            all_requests.append((task,directory,phase,row))
            if arm=='sgl':
                assert str(request['profile']).lower()=='false'
                perf=json.loads((directory/'perf.json').read_text())
                assert perf['commit_hash']==spec['head'] and len(perf['denoise_steps_ms'])==50
                durations=[s['duration_ms']/1000 for s in perf['steps'] if s['name']=='QwenImage21DenoisingStage']
                assert len(durations)==1 and 0<durations[0]<row['http_seconds']
                if not row['warmup']:denoise.append(durations[0])
        task_results[task]=dict(denoise_seconds=denoise,denoise_p50_seconds=statistics.median(denoise) if denoise else None)
    correlations=[]
    if arm=='omni':
        raw_log=(server/'server.log').read_bytes()
        log=re.sub(r'\x1b\[[0-9;]*m','',raw_log.decode(errors='replace'))
        matches=list(re.finditer(r'\(DiffusionWorker pid=(\d+)\).*?\[DiffusionPipelineProfiler\] QwenImage21Pipeline\.diffuse took ([0-9.]+)s',log))
        pids={int(m[1]) for m in matches}
        assert len(pids)==1
        pid=pids.pop()
        observed=[json.loads(line) for line in (ROOT/'evidence/diagnostics'/name/f'omni-worker-{pid}.jsonl').read_text().splitlines()]
        assert len(observed)==len(matches)
        startup_count=len(observed)-len(all_requests)
        assert startup_count>=1
        boundary=log.index('Application startup complete.')
        assert matches[startup_count-1].end()<boundary<matches[startup_count].start()
        assert all(row['hook_sha256']==plan['stage_observer_sha256'] and row['pid']==pid for row in observed)
        assert all(a['finished_ns']<b['started_ns'] for a,b in zip(observed,observed[1:]))
        for i in range(startup_count):
            correlations.append(dict(phase='native_startup',scheduler_steps=len(observed[i]['scheduler_calls']),native_diffuse_seconds=float(matches[i][2]),excluded=True))
        for i,(task,directory,phase,row) in enumerate(all_requests,startup_count):
            record=observed[i]
            assert len(record['scheduler_calls'])==50
            value=float(matches[i][2])
            response=json.loads((directory/'response.json').read_text())
            gap=response['created']-record['finished_ns']/1e9
            assert -1<=gap<=5 and 0<value<row['http_seconds']
            assert (record['finished_ns']-record['started_ns'])/1e9>=value-.001
            if phase=='diagnostic':task_results[task]['denoise_seconds'].append(value)
            correlations.append(dict(task=task,phase=phase,request=str(directory.relative_to(ROOT)),scheduler_steps=50,native_diffuse_seconds=value,response_created_gap_seconds=gap,output_sha256=row['sha256']))
        for task,values in task_results.items():
            assert len(values['denoise_seconds'])==plan['repeats']
            values['denoise_p50_seconds']=statistics.median(values['denoise_seconds'])
    checks[config]=dict(tasks=task_results,correlations=correlations,source_sha=spec['head'],raw_audit_pass=True)
out=ROOT/'evidence/preflight'/f'{run}-audit.json'
assert not out.exists()
out.write_text(json.dumps(dict(primary_timing=False,native_stage_timing=True,torch_profiler_during_measured_requests=False,all_raw_audits_pass=True,checks=checks,caveat='Stage scopes are native framework boundaries. SGL adds stage/step synchronization; Omni wraps diffuse with its native timer and a metadata-only Python observer. Both are excluded from primary E2E ratios. Separate Torch profile requests are audited independently.'),indent=2)+'\n')
print(json.dumps({config:check['tasks'] for config,check in checks.items()}))
