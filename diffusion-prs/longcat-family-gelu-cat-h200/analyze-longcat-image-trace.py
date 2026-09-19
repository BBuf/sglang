"""Summarize an LongCat T2I profile and a complete two-pass CFG iteration."""
import collections
import gzip
import json
from pathlib import Path
import sys

source=Path(sys.argv[1])
with gzip.open(source,'rt') as stream:
    data=json.load(stream)
events=data['traceEvents']
models=sorted((e for e in events if e.get('name')=='nn.Module: LongCatImageTransformer2DModel_0'),key=lambda e:e['ts'])
graphs=sorted((e for e in events if e.get('name')=='cudaGraphLaunch'),key=lambda e:e['ts'])
if models:
    assert len(models)==8,len(models)
    lo,hi=models[2]['ts'],models[4]['ts']
    boundary='third to fifth model start: complete second CFG iteration (positive and negative passes), native synchronization enabled'
else:
    # Native log records 31 captured segments and the profile records eight model calls across four CFG iterations.
    assert len(graphs)==8*31,len(graphs)
    lo,hi=graphs[2*31]['ts'],graphs[4*31]['ts']
    boundary='third to fifth graph-replayed model call (complete CFG iteration) (31 graph segments per call); stage synchronization enabled'
assert not [e for e in events if e.get('cat')=='kernel' and any(e['ts']<t<e['ts']+e['dur'] for t in (lo,hi))]
selected=[]
for e in events:
    ts=e.get('ts')
    if e.get('ph')=='M': selected.append(e)
    elif ts is not None and lo<=ts<hi:
        if e.get('ph')=='X' and ts+e.get('dur',0)>hi:
            e=e|{'dur':hi-ts,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}}
        selected.append(e)
    elif ts is not None and e.get('ph')=='X' and ts<lo<ts+e.get('dur',0) and e.get('cat')!='kernel':
        selected.append(e|{'ts':lo,'dur':min(hi,ts+e['dur'])-lo,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}})
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
            start_us=lo,end_us=hi,crossing_kernels=0,window_ms=window,gpu_union_ms=busy,
            uncovered_interval_ms=window-busy,uncovered_interval_pct=(window-busy)/window*100,
            kernel_count=sum(v[0] for v in rows.values()),
            graph_launch_api_count=sum(e.get('name')=='cudaGraphLaunch' for e in selected),
            launch_api_count=sum(e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name','') for e in selected),
            top_kernels=[dict(name=k,count=v[0],total_ms=v[1]) for k,v in sorted(rows.items(),key=lambda kv:-kv[1][1])[:35]],
            note='Uncovered GPU intervals suggest a launch-overhead opportunity; they are not a promise of recoverable E2E time.')
target.with_name('denoise-step2-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
