"""Inspect native Cosmos3 traces and isolate one complete GEN forward by launch ownership."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import sys

source=Path(sys.argv[1])
with gzip.open(source,'rt') as f: data=json.load(f)
events=data['traceEvents']
models=sorted((e for e in events if e.get('name')=='nn.Module: Cosmos3OmniTransformer_0'),key=lambda e:e['ts'])
assert len(models)>=3, len(models)
launches=sorted((e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{}) and ('LaunchKernel' in e.get('name','') or 'LaunchCooperativeKernel' in e.get('name',''))),key=lambda e:e['ts'])
launch_times=[e['ts'] for e in launches]
kernels=collections.defaultdict(list)
for e in events:
 if e.get('cat')=='kernel': kernels[e.get('args',{}).get('correlation')].append(e)
def owned(scope):
 lo=bisect.bisect_left(launch_times,scope['ts']); hi=bisect.bisect_left(launch_times,scope['ts']+scope['dur'])
 return [k for launch in launches[lo:hi] for k in kernels[launch['args']['correlation']]]
def summary(ks):
 d=collections.defaultdict(lambda:[0,0.0])
 for k in ks: d[k['name']][0]+=1; d[k['name']][1]+=k['dur']/1000
 return [dict(name=k,count=v[0],gpu_ms=v[1]) for k,v in sorted(d.items(),key=lambda kv:-kv[1][1])]
rows=[]
for idx,m in enumerate(models):
 ks=owned(m)
 rows.append(dict(index=idx,cpu_us=m['dur'],kernel_count=len(ks),gpu_ms=sum(k['dur'] for k in ks)/1000,start_us=m['ts'],end_us=m['ts']+m['dur']))
selected_model=models[2]; ks=owned(selected_model); assert ks
corr={k['args']['correlation'] for k in ks}; cpu_lo=selected_model['ts'];cpu_hi=cpu_lo+selected_model['dur']
lo=min(k['ts'] for k in ks);hi=max(k['ts']+k['dur'] for k in ks)
selected=[]
for e in events:
 if e.get('ph')=='M': selected.append(e);continue
 ts=e.get('ts');cat=e.get('cat')
 if ts is None:continue
 if cat=='kernel':
  if e.get('args',{}).get('correlation') in corr:selected.append(e)
 elif cat in ('gpu_memcpy','gpu_memset'):
  if e.get('args',{}).get('correlation') in corr:selected.append(e)
 elif cpu_lo<=ts<cpu_hi:
  if e.get('ph')=='X' and ts+e.get('dur',0)>cpu_hi:e=e|{'dur':cpu_hi-ts}
  selected.append(e)
 elif e.get('ph')=='X' and ts<cpu_lo<ts+e.get('dur',0):
  selected.append(e|{'ts':cpu_lo,'dur':min(cpu_hi,ts+e['dur'])-cpu_lo})
target=source.with_name('forward3.trace.json.gz')
with gzip.open(target,'wt',compresslevel=1) as f:json.dump(data|{'traceEvents':selected},f,separators=(',',':'))
scopes=collections.defaultdict(list)
for e in events:
 if cpu_lo<=e.get('ts',-1)<cpu_hi and e.get('name','').startswith('nn.Module: '):scopes[e['name']].append(e)
module_rows=[]
for name,ss in scopes.items():
 child=[k for scope in ss for k in owned(scope)]
 module_rows.append(dict(name=name,calls=len(ss),kernels=len(child),gpu_ms=sum(k['dur'] for k in child)/1000))
intervals=sorted((k['ts'],k['ts']+k['dur']) for k in ks); merged=[]
for a,b in intervals:
 if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
 else:merged.append([a,b])
report=dict(source=str(source),model_calls=len(models),forwards=rows,selected_forward_index=2,kernel_count=len(ks),gpu_window_ms=(hi-lo)/1000,gpu_union_ms=sum(b-a for a,b in merged)/1000,top_kernels=summary(ks),module_scopes=sorted(module_rows,key=lambda r:-r['gpu_ms']),note='Third complete native model call. Kernel ownership uses CUDA launch correlation; overlapping parent/child module times must not be summed. No CFG count assumed.')
target.with_name('forward3-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('forwards','top_kernels','module_scopes')}))
print(json.dumps(report['forwards'][:5],indent=2))
print(json.dumps(report['top_kernels'][:20],indent=2))
