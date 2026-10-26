"""Summarize GPU kernels inside complete profiled denoise step annotations."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('trace',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
events=json.load(gzip.open(a.trace))['traceEvents']
kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']

def summarize(items):
    grouped=collections.defaultdict(lambda:[0,0.0])
    for e in items:
        row=grouped[e['name']];row[0]+=1;row[1]+=e['dur']/1000
    return [dict(name=n,calls=v[0],sum_ms=v[1]) for n,v in sorted(grouped.items(),key=lambda kv:-kv[1][1])]

steps=[]
for e in events:
    if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('ProfilerStep#'):
        inside=[k for k in kernels if e['ts']<=k['ts'] and k['ts']+k['dur']<=e['ts']+e['dur']+1]
        steps.append(dict(name=e['name'],gpu_annotation_ms=e['dur']/1000,kernel_count=len(inside),
                          kernel_sum_ms=sum(k['dur'] for k in inside)/1000,kernels=summarize(inside)))
r=dict(trace=str(a.trace),trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
       caveat=f'Profile durations are diagnostic, not profiler-free latency. Requested3steps; trace contains{len(steps)}GPUstep annotations. Compare matching complete step numbers, not whole-trace sums.',
       all_kernel_count=len(kernels),all_kernel_sum_ms=sum(k['dur'] for k in kernels)/1000,
       cuda_graph_launch_count=sum(e.get('name')=='cudaGraphLaunch' for e in events),
       steps=sorted(steps,key=lambda x:x['name']),all_kernels=summarize(kernels))
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ('steps','all_kernels')}))
for s in r['steps']:
    copies=[k for k in s['kernels'] if 'copy' in k['name'].lower() and 'bfloat16' in k['name'].lower()]
    print(s['name'],'GPU annotation ms',s['gpu_annotation_ms'],'kernel sum',s['kernel_sum_ms'],'BF16 copies',sum(x['calls'] for x in copies),sum(x['sum_ms'] for x in copies))
