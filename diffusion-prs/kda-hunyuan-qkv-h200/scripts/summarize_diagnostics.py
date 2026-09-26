#!/usr/bin/env python3
"""Summarize diagnostic calls and actual CUDA symbols without implying E2E gains."""
import argparse
import bisect
from collections import Counter
import gzip
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
result={'diagnostic_only':True,'directory':str(a.directory)}
counts=Counter()
for path in (a.directory/'calls').glob('calls-*.jsonl'):
 for line in path.open():
  row=json.loads(line)
  counts[json.dumps(row,sort_keys=True)]+=1
result['python_calls']=[dict(json.loads(key),count=count) for key,count in counts.items()]
result['traces']=[]
for path in (a.directory/'traces').glob('*.trace.json*'):
 opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt') as f:events=json.load(f)['traceEvents']
 kernel_counts=Counter();kernel_us=Counter()
 for event in events:
  if event.get('cat')=='kernel':
   kernel_counts[event['name']]+=1
   kernel_us[event['name']]+=event.get('dur',0)
 kernels=[dict(name=name,count=kernel_counts[name],cumulative_gpu_us=kernel_us[name]) for name in kernel_counts if any(text in name.lower() for text in ['rmsnorm','tanh_residual','cat_pad','nearest_upsample','ada_values','_gn_','group_norm','bias_silu','bias_glu','silu_mul','complex_rope','rotate_half','joint_qkv','hunyuan','qkvs'])]
 scopes=Counter()
 cpu=sorted((event for event in events if event.get('cat')=='cpu_op'),key=lambda event:event['ts'])
 times=[event['ts'] for event in cpu]
 for event in events:
  if event.get('cat')=='python_function' and any(name in event['name'] for name in ['): group_norm_silu_4d','): group_norm_silu_rows','): triton_group_norm_silu']):
   low=bisect.bisect_left(times,event['ts']);high=bisect.bisect_left(times,event['ts']+event['dur'])
   ops=[dict(name=op['name'],shapes=op.get('args',{}).get('Input Dims')) for op in cpu[low:high] if op.get('pid')==event.get('pid') and op.get('tid')==event.get('tid') and op.get('args',{}).get('Input Dims')]
   scopes[json.dumps(dict(function=event['name'],first_cpu_ops=ops[:5]),sort_keys=True)]+=1
 result['traces'].append(dict(path=str(path),kernel_count=sum(kernel_counts.values()),selected_kernels=kernels,groupnorm_python_scopes=[dict(json.loads(key),count=count) for key,count in scopes.items()]))
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(call_signatures=len(result['python_calls']),trace_count=len(result['traces']),output=str(a.output))))
