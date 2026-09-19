"""Summarize an Cosmos3 Edge profile and a complete two-pass CFG iteration."""
import collections
import gzip
import json
from pathlib import Path
import sys

source=Path(sys.argv[1])
with gzip.open(source,'rt') as stream:
    data=json.load(stream)
events=data['traceEvents']
models=sorted((e for e in events if e.get('name')=='nn.Module: Cosmos3OmniTransformer_0'),key=lambda e:e['ts'])
graphs=sorted((e for e in events if e.get('name')=='cudaGraphLaunch'),key=lambda e:e['ts'])
if models:
    assert len(models)==8,len(models)
    lo,hi=models[2]['ts'],models[4]['ts']
    boundary='third to fifth model start: complete second CFG iteration (positive and negative passes), native synchronization enabled'
else:
    # Native log records 31 captured segments and the request has 8 steps.
    assert len(graphs)==8*31,len(graphs)
    lo,hi=graphs[2*31]['ts'],graphs[3*31]['ts']
    boundary='third to fourth graph-replayed model call (31 graph segments per call); stage synchronization enabled'
cpu_lo, cpu_hi = lo, hi
# CUDA work may trail the host model call. Correlate complete CFG CPU ranges
# to GPU kernels instead of cutting a busy GPU at a host timestamp.
ids = {e['args']['correlation'] for e in events
       if e.get('cat') in ('cpu_op', 'cuda_runtime', 'cuda_driver')
       and cpu_lo <= e.get('ts', -1) < cpu_hi
       and 'correlation' in e.get('args', {})}
next_ids = {e['args']['correlation'] for e in events
            if e.get('cat') in ('cpu_op', 'cuda_runtime', 'cuda_driver')
            and cpu_hi <= e.get('ts', -1)
            and 'correlation' in e.get('args', {})}
cycle_kernels = [e for e in events if e.get('cat') == 'kernel'
                 and e.get('args', {}).get('correlation') in ids]
next_kernels = [e for e in events if e.get('cat') == 'kernel'
                and e.get('args', {}).get('correlation') in next_ids]
lo = min(e['ts'] for e in cycle_kernels)
hi = min(e['ts'] for e in next_kernels)
assert max(e['ts'] + e['dur'] for e in cycle_kernels) <= hi
assert not [e for e in events if e.get('cat') == 'kernel' and any(e['ts'] < t < e['ts'] + e['dur'] for t in (lo, hi))]
assert all(e.get('args', {}).get('correlation') in ids for e in events
           if e.get('cat') == 'kernel' and lo <= e['ts'] < hi)
boundary = 'Complete second CFG iteration; CPU third-to-fifth model starts joined to GPU kernels by CUDA launch correlation, GPU first-kernel to next-iteration first-kernel boundaries.' 
selected=[]
for e in events:
    ts=e.get('ts')
    if e.get('ph')=='M':
        selected.append(e)
        continue
    if ts is None:
        continue
    gpu = e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')
    start,end = (lo,hi) if gpu else (cpu_lo,cpu_hi)
    if start <= ts < end:
        if not gpu and e.get('ph')=='X' and ts+e.get('dur',0)>end:
            e=e|{'dur':end-ts,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}}
        selected.append(e)
    elif not gpu and e.get('ph')=='X' and ts<start<ts+e.get('dur',0):
        selected.append(e|{'ts':start,'dur':min(end,ts+e['dur'])-start,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}})
target=source.with_name('denoise-step2.trace.json.gz')
with gzip.open(target,'wt',compresslevel=1) as stream:
    json.dump(data|{'traceEvents':selected},stream,separators=(',',':'))
rows=collections.defaultdict(lambda:[0,0.0])
for e in selected:
    if e.get('cat')=='kernel':
        rows[e['name']][0]+=1
        rows[e['name']][1]+=e['dur']/1000
intervals=sorted((max(lo,e['ts']),min(hi,e['ts']+e['dur'])) for e in selected
                 if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('dur',0)>0)
merged=[]
for start,end in intervals:
    if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
    else: merged.append([start,end])
busy=sum(end-start for start,end in merged)/1000
window=(hi-lo)/1000
report=dict(source=str(source),source_events=len(events),slice=str(target),slice_events=len(selected),
            model_calls=len(models) or len(graphs)//31,boundary=boundary,
            start_us=lo,end_us=hi,cpu_start_us=cpu_lo,cpu_end_us=cpu_hi,crossing_kernels=0,window_ms=window,gpu_union_ms=busy,
            uncovered_interval_ms=window-busy,uncovered_interval_pct=(window-busy)/window*100,
            kernel_count=sum(v[0] for v in rows.values()),
            graph_launch_api_count=sum(e.get('name')=='cudaGraphLaunch' for e in selected),
            launch_api_count=sum(e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name','') for e in selected),
            top_kernels=[dict(name=k,count=v[0],total_ms=v[1]) for k,v in sorted(rows.items(),key=lambda kv:-kv[1][1])[:35]],
            note='Uncovered GPU intervals suggest a launch-overhead opportunity; they are not a promise of recoverable E2E time.')
target.with_name('denoise-step2-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
