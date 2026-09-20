"""Join Wan VAE norm/SiLU module scopes to GPU kernels and native input shapes."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import sys

path=Path(sys.argv[1])
with gzip.open(path,'rt') as f:events=json.load(f)['traceEvents']
scopes=sorted((e for e in events if e.get('name','').startswith('nn.Module: FusedWanRMSNormSiLU_')),key=lambda e:e['ts'])
assert scopes
starts=[e['ts'] for e in scopes]
def owner(e):
    i=bisect.bisect_right(starts,e.get('ts',-1))-1
    return i if i>=0 and e['ts']<scopes[i]['ts']+scopes[i]['dur'] else None
launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name','') and 'correlation' in e.get('args',{})}
rows=collections.defaultdict(lambda:[0,0.0]);ops=collections.defaultdict(int)
for e in events:
    if e.get('cat')=='cpu_op' and owner(e) is not None:
        args=e.get('args',{});ops[(e['name'],json.dumps(args.get('Input Dims')),json.dumps(args.get('Input type')))]+=1
    if e.get('cat')=='kernel':
        launch=launches.get(e.get('args',{}).get('correlation'))
        if launch is not None and owner(launch) is not None:
            rows[e['name']][0]+=1;rows[e['name']][1]+=e.get('dur',0)/1000
report=dict(source=str(path),scope_count=len(scopes),kernel_count=sum(v[0] for v in rows.values()),gpu_ms=sum(v[1] for v in rows.values()),
    kernels=[dict(name=k,count=v[0],gpu_ms=v[1]) for k,v in sorted(rows.items(),key=lambda kv:-kv[1][1])],
    ops=[dict(name=k[0],input_dims=json.loads(k[1]),input_types=json.loads(k[2]),count=v) for k,v in sorted(ops.items(),key=lambda kv:-kv[1])])
path.with_name('wan-norm-silu-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='ops'},indent=2))
