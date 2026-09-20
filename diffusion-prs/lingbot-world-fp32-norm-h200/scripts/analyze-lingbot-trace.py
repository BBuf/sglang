"""Attribute complete LingBot forwards and their module sites using CUDA correlations."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import sys
path=Path(sys.argv[1]);events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
families=['CausalLingBotWorldTransformer3DModel','CausalLingBotWorldTransformerBlock','FP32LayerNorm','LingBotWorldCamConditioner','ScaleResidualLayerNormScaleShift','MLP','MulAdd','LingBotWorldCausalSelfAttention','USPAttention','RMSNorm']
scopes={f:sorted([e for e in events if e.get('name','').startswith('nn.Module: '+f+'_')],key=lambda e:e['ts']) for f in families}
starts={f:[e['ts'] for e in ss] for f,ss in scopes.items()}
def owner(e,f):
 i=bisect.bisect_right(starts[f],e.get('ts',-1))-1
 if i>=0:
  s=scopes[f][i]
  if e['ts']<s['ts']+s['dur'] and e.get('tid')==s.get('tid'):return i
 return None
launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name','') and 'correlation' in e.get('args',{})}
kernels=[e for e in events if e.get('cat')=='kernel'];rows=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0.]));ops=collections.defaultdict(collections.Counter);passes=collections.defaultdict(list)
for e in kernels:
 launch=launches.get(e.get('args',{}).get('correlation'))
 if launch is None or owner(launch,families[0]) is None:continue
 ix=owner(launch,families[0]);passes[ix].append(e)
 matched=[f for f in families[1:] if owner(launch,f) is not None]
 f=min(matched,key=lambda f:scopes[f][owner(launch,f)]['dur']) if matched else 'model-outside-block'
 rows[f][e['name']][0]+=1;rows[f][e['name']][1]+=e['dur']/1000
for e in events:
 if e.get('cat')!='cpu_op' or owner(e,families[1]) is None:continue
 matched=[f for f in families[2:] if owner(e,f) is not None]
 f=min(matched,key=lambda f:scopes[f][owner(e,f)]['dur']) if matched else 'block-direct'
 a=e.get('args',{});ops[f][(e['name'],json.dumps(a.get('Input Dims')),json.dumps(a.get('Input type')))]+=1
report=dict(source=str(path),scope_counts={f:len(s) for f,s in scopes.items()},families={},passes=[])
for f,ks in rows.items():
 report['families'][f]=dict(kernels=sum(v[0] for v in ks.values()),gpu_ms=sum(v[1] for v in ks.values()),kernel_types=[dict(name=k,count=v[0],gpu_ms=v[1]) for k,v in sorted(ks.items(),key=lambda kv:-kv[1][1])])
for f,rs in ops.items():
 report['families'].setdefault(f,{})['ops']=[dict(name=k[0],shapes=json.loads(k[1]),dtypes=json.loads(k[2]),count=v) for k,v in rs.most_common()]
for ix,ks in sorted(passes.items()):
 lo=min(e['ts'] for e in ks);hi=max(e['ts']+e['dur'] for e in ks);ids={id(e) for e in ks}
 crossings=[e for e in kernels if id(e) not in ids and e['ts']<hi and e['ts']+e['dur']>lo]
 r=dict(index=ix,launch_count=len(ks),gpu_ms=sum(e['dur'] for e in ks)/1000,wall_ms=(hi-lo)/1000,other_kernels_in_gpu_span=len(crossings),cpu_scope_us=[scopes[families[0]][ix]['ts'],scopes[families[0]][ix]['dur']])
 report['passes'].append(r)
 if ix==2:
  # GPU kernels are selected by their CPU launch ownership, not by the
  # CPU submission time window, which can overlap another forward's work.
  cscope=scopes[families[0]][ix]
  sliced=[e for e in events if e.get('cat')=='kernel' and id(e) in ids]
  sliced += [e for e in events if e.get('cat')!='kernel' and (
      'ts' not in e or (e.get('tid')==cscope.get('tid') and
      cscope['ts']<=e.get('ts',-1)<cscope['ts']+cscope['dur']))]
  with gzip.open(path.with_name('forward3.trace.json.gz'),'wt') as out:json.dump(dict(traceEvents=sliced),out)
path.with_name('lingbot-attribution.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(scopes=report['scope_counts'],passes=report['passes'],families={f:{k:v for k,v in r.items() if k in ('kernels','gpu_ms')} for f,r in report['families'].items()}),indent=2))

# Compare four non-overlapping portions of each block using CPU launch
# correlations. Include entry/affine setup in norm1; do not mistake these
# cumulative GPU kernel times for unprofiled request latency.
by_block=collections.defaultdict(list)
for e in kernels:
 launch=launches.get(e.get('args',{}).get('correlation'))
 if launch is None:continue
 ix=owner(launch,'CausalLingBotWorldTransformerBlock')
 if ix is not None:by_block[ix].append((launch,e))
site_rows=collections.defaultdict(list)
for ix,pairs in sorted(by_block.items()):
 block=scopes['CausalLingBotWorldTransformerBlock'][ix]
 pairs.sort(key=lambda p:p[0]['ts'])
 gemms=[p for p in pairs if 'gemm' in p[1]['name'].lower() or p[1]['name'].startswith('nvjet_')]
 assert gemms,(ix,'No QKV GEMM boundary')
 cameras=[s for s in scopes['LingBotWorldCamConditioner'] if s['ts']>=block['ts'] and s['ts']+s['dur']<=block['ts']+block['dur'] and s['tid']==block['tid']]
 assert len(cameras)<=1,(ix,len(cameras))
 ranges={'norm1_and_entry_setup':(block['ts'],gemms[0][0]['ts'])}
 if cameras:
  camera=cameras[0];lo=camera['ts'];hi=lo+camera['dur']
  attentions=[s for s in scopes['LingBotWorldCausalSelfAttention'] if block['ts']<=s['ts'] and s['ts']+s['dur']<=lo and s['tid']==block['tid']]
  assert len(attentions)==1,(ix,len(attentions))
  attn_end=attentions[0]['ts']+attentions[0]['dur']
  projection=next(p for p in gemms if attn_end<=p[0]['ts']<lo)
  residual_start=projection[0]['ts']+projection[0]['dur']
  camera_mlps=[s for s in scopes['MLP'] if residual_start<=s['ts']<lo and s['tid']==block['tid']]
  residual_end=min([lo]+[s['ts'] for s in camera_mlps])
  following=[p for p in gemms if p[0]['ts']>=hi]
  assert following,ix
  ranges.update(residual_before_camera=(residual_start,residual_end),camera=(lo,hi),cross_attention_norm=(hi,following[0][0]['ts']))
 for site,(lo,hi) in ranges.items():
  selected=[e for launch,e in pairs if lo<=launch['ts']<hi]
  grouped=collections.defaultdict(lambda:[0,0.])
  for e in selected:grouped[e['name']][0]+=1;grouped[e['name']][1]+=e['dur']/1000
  site_rows[site].append(dict(block=ix,cpu_interval_us=[lo,hi],kernels=len(selected),gpu_ms=sum(e['dur'] for e in selected)/1000,kernel_types=[dict(name=k,count=v[0],gpu_ms=v[1]) for k,v in grouped.items()]))
site_report={}
for site,rs in site_rows.items():
 grouped=collections.defaultdict(lambda:[0,0.])
 for r in rs:
  for k in r['kernel_types']:grouped[k['name']][0]+=k['count'];grouped[k['name']][1]+=k['gpu_ms']
 site_report[site]=dict(blocks=len(rs),kernels=sum(r['kernels'] for r in rs),gpu_ms=sum(r['gpu_ms'] for r in rs),kernel_types=[dict(name=k,count=v[0],gpu_ms=v[1]) for k,v in sorted(grouped.items(),key=lambda kv:-kv[1][1])],per_block=rs)
path.with_name('lingbot-sites.json').write_text(json.dumps(dict(source=str(path),sites=site_report,definition='Non-overlapping CPU launch intervals inside complete causal blocks; norm1 includes block-entry affine setup; camera is its module scope; residual starts after the first projection following self-attention and ends before camera-affine MLP preparation or the camera module; cross norm ends before its first projection GEMM. GPU kernel time only, excluding memcpys and host time.'),indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a not in ('kernel_types','per_block')} for k,v in site_report.items()},indent=2))
