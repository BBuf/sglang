"""Extract a complete denoising cycle, bounded by matching completed host syncs."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import sys

source=Path(sys.argv[1])
with gzip.open(source,'rt') as stream:
    data=json.load(stream)
events=data['traceEvents']
models=sorted((e for e in events if e.get('name')=='nn.Module: Qwen3Model_0'),key=lambda e:e['ts'])
assert len(models)==102,len(models) # two text prefixes + two CFG passes per step
def boundary(module):
    matches=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('name')=='cudaStreamSynchronize'
             and module['ts']<=e.get('ts',-1)<module['ts']+module['dur']]
    first=min(matches,key=lambda e:e['ts'])
    return first['ts']+first['dur'],first
start,left=boundary(models[6])
end,right=boundary(models[8])
crossing=[e for e in events if e.get('cat')=='kernel' and any(e['ts']<t<e['ts']+e.get('dur',0) for t in (start,end))]
assert not crossing,crossing[:3]
selected=[]
for e in events:
    ts=e.get('ts')
    if e.get('ph')=='M':
        selected.append(e)
    elif ts is not None and start<=ts<end:
        if e.get('ph')=='X' and ts+e.get('dur',0)>end:
            e=e|{'dur':end-ts,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}}
        selected.append(e)
    elif ts is not None and e.get('ph')=='X' and ts<start<ts+e.get('dur',0) and e.get('cat')!='kernel':
        selected.append(e|{'ts':start,'dur':min(end,ts+e['dur'])-start,'args':e.get('args',{})|{'campaign_clipped_cpu_span':True}})
destination=source.with_name('denoise-step2.trace.json.gz')
with gzip.open(destination,'wt',compresslevel=1) as stream:
    json.dump(data|{'traceEvents':selected},stream,separators=(',',':'))
ropes=sorted((e for e in selected if e.get('cat')=='python_function' and 'modeling_qwen3.py' in e.get('name','') and 'apply_rotary_pos_emb' in e['name']),key=lambda e:e['ts'])
starts=[e['ts'] for e in ropes]
launches={e.get('args',{}).get('correlation'):e for e in selected if e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name','')}
attributed=collections.defaultdict(lambda:[0,0.0])
for e in selected:
    if e.get('cat')!='kernel': continue
    launch=launches.get(e.get('args',{}).get('correlation'))
    if not launch: continue
    i=bisect.bisect_right(starts,launch['ts'])-1
    if i>=0 and launch['ts']<ropes[i]['ts']+ropes[i]['dur']:
        attributed[e['name']][0]+=1
        attributed[e['name']][1]+=e['dur']/1000
report=dict(source=str(source),slice=str(destination),source_event_count=len(events),slice_event_count=len(selected),
    start_us=start,end_us=end,window_ms=(end-start)/1000,boundary_rule='First completed cudaStreamSynchronize in Qwen3Model forwards 6 and 8; two prefixes precede 50 pairs of CFG forwards',
    left_boundary=left,right_boundary=right,crossing_kernel_count=0,rope_calls=len(ropes),
    rope_kernel_count=sum(v[0] for v in attributed.values()),rope_gpu_ms=sum(v[1] for v in attributed.values()),
    rope_kernels=[dict(name=k,count=v[0],total_ms=v[1]) for k,v in sorted(attributed.items(),key=lambda kv:-kv[1][1])])
destination.with_name('denoise-step2-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
