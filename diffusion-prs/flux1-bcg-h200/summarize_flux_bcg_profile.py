"""Union GPU stage intervals and retain complete operator/kernel evidence per rank."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

root = Path('/data/goals/sglang-h200-diffusion-20260907')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cell', required=True)
args = parser.parse_args()
cell = root/'results'/args.cell

def aggregate(events):
    result = defaultdict(lambda: dict(count=0, duration_ms=0.0))
    for e in events:
        row = result[e['name']]
        row['count'] += 1
        row['duration_ms'] += e.get('dur',0)/1000
    return [dict(name=k,**v) for k,v in sorted(result.items(),key=lambda x:x[1]['duration_ms'],reverse=True)]

reports=[]
for path in sorted((cell/'profiler').glob('*.json*')):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream: data=json.load(stream)
    events=data['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel']
    grouped=defaultdict(list)
    for e in events:
        if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('STAGE '):
            grouped[(e['name'],e['pid'])].append((e['ts'],e['ts']+e['dur']))
    stages=[]
    for (name,pid),ranges in grouped.items():
        union=[]
        for lo,hi in sorted(ranges):
            if union and lo<=union[-1][1]:union[-1][1]=max(union[-1][1],hi)
            else:union.append([lo,hi])
        selected=[e for e in kernels if e['pid']==pid and any(lo<=e['ts']<hi for lo,hi in union)]
        crossing=[e for e in selected if not any(lo<=e['ts'] and e['ts']+e['dur']<=hi+.01 for lo,hi in union)]
        copies=[e for e in events if e.get('cat')=='gpu_memcpy' and e['pid']==pid and any(lo<=e['ts']<hi for lo,hi in union)]
        stages.append(dict(name=name,gpu_pid=pid,annotation_count=len(ranges),union_ms=sum(hi-lo for lo,hi in union)/1000,
                           kernel_count=len(selected),cumulative_kernel_ms=sum(e['dur'] for e in selected)/1000,
                           boundary_crossing_kernel_count=len(crossing),kernels=aggregate(selected),copies=aggregate(copies)))
    cpu_stages=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('STAGE ')]
    cpu_report=[]
    for scope in cpu_stages:
        ops=[e for e in events if e.get('cat')=='cpu_op' and e['pid']==scope['pid'] and scope['ts']<=e['ts']<scope['ts']+scope['dur']]
        by_shape=defaultdict(lambda:dict(count=0,duration_ms=0.0))
        for e in ops:
            key=(e['name'],json.dumps(e.get('args',{}).get('Input Dims',[])))
            by_shape[key]['count']+=1;by_shape[key]['duration_ms']+=e['dur']/1000
        runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e['pid']==scope['pid'] and scope['ts']<=e['ts']<scope['ts']+scope['dur']]
        cpu_report.append(dict(name=scope['name'],duration_ms=scope['dur']/1000,ops=aggregate(ops),cuda_apis=aggregate(runtime),
                               shapes=[dict(name=k[0],input_dims=json.loads(k[1]),**v) for k,v in sorted(by_shape.items(),key=lambda x:x[1]['duration_ms'],reverse=True)]))
    reports.append(dict(trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                        events=len(events),kernel_count=len(kernels),gpu_stages=stages,cpu_stages=cpu_report))
assert reports
output=root/'results'/f'{args.cell}-digest.json'
output.write_text(json.dumps(dict(scope='Full two-step diagnostic, each rank separately. Overlapping GPU stage annotations unioned; CPU op durations can nest and are not additive. Profile wall time excludes no export overhead and is not E2E benchmark evidence.',traces=reports),indent=2)+'\n')
for report in reports:
    print(report['trace'])
    for stage in report['gpu_stages']:
        print(stage['name'],stage['kernel_count'],stage['cumulative_kernel_ms'],'ms','crossing',stage['boundary_crossing_kernel_count'])
        if 'denois' in stage['name'].lower():print(json.dumps(stage['kernels'][:18],indent=2))
