"""Matched full third-step GPU anchors and actual MLP shape/dispatch evidence."""
import collections,gzip,hashlib,json
from pathlib import Path
from analyze_joy_gelu_profiles import aggregate,inside
ROOT=Path(__file__).resolve().parents[1]
def analyze(path):
 raw=path.read_bytes();events=json.loads(gzip.decompress(raw))['traceEvents'];name='ProfilerStep#3'
 ca=[e for e in events if e.get('cat')=='user_annotation' and e['name']==name];ga=[e for e in events if e.get('cat')=='gpu_user_annotation' and e['name']==name];assert len(ca)==len(ga)==1;ca,ga=ca[0],ga[0];assert ca['args']['External id']==ga['args']['External id']
 cpu=[e for e in events if e.get('cat')=='cpu_op' and inside(e,ca)];gpu=[e for e in events if e.get('cat')=='kernel' and inside(e,ga)]
 shapes=collections.Counter(json.dumps(e.get('args',{}).get('Input Dims')) for e in cpu if e['name']=='aten::mm')
 return dict(trace=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),step=name,gpu_annotation_ms=ga['dur']/1000,cpu_annotation_ms=ca['dur']/1000,kernel_count=len(gpu),kernel_sum_ms=sum(e['dur'] for e in gpu)/1000,kernels=aggregate(gpu),mm_shapes=[dict(input_dims=json.loads(s),calls=n) for s,n in shapes.items()],graph_launches=sum(e.get('name')=='cudaGraphLaunch' and inside(e,ca) for e in events))
rows={}
for task in ['t2i','edit']:
 arms={}
 for label,qual,profile in [('main','mainhigh1','qwen21-base-main-high-qualification1-sgl'),('candidate','gate-up-v1','qwen21-gate-up-v1-profile1-sgl')]:
  d=ROOT/'evidence/qualification'/f'sgl-qwen21-base-{task}-{qual}-profile';perf=json.loads(next(d.glob('*/perf.json')).read_text());trace=list((ROOT/'evidence/profiles'/profile).glob(perf['request_id']+'*.gz'));assert len(trace)==1;arms[label]=analyze(trace[0])
 m,c=arms['main'],arms['candidate'];checks=dict(main_separate_gate_up=sum(x['calls'] for x in m['mm_shapes'] if x['input_dims']==[[4096,4096],[4096,12288]]),candidate_packed_gate_up=sum(x['calls'] for x in c['mm_shapes'] if x['input_dims']==[[4096,4096],[4096,24576]]),candidate_packed_silu=sum(x['calls'] for x in c['kernels'] if '_packed_silu_mul_kernel' in x['name']),kernel_count_reduction=m['kernel_count']-c['kernel_count'])
 checks['dispatch_pass']=(checks['main_separate_gate_up']==64 and checks['candidate_packed_gate_up']==32 and checks['candidate_packed_silu']==32 and checks['kernel_count_reduction']==32)
 rows[task]=dict(**arms,checks=checks)
 print(task,checks,'GPU step ms',m['gpu_annotation_ms'],c['gpu_annotation_ms'])
out=ROOT/'evidence/profiles/qwen21-gate-up-v1-matched-step3.json';assert not out.exists();out.write_text(json.dumps(dict(tasks=rows,all_dispatch_pass=all(v['checks']['dispatch_pass'] for v in rows.values()),caveat='Diagnostic profiles, no primary E2E or synchronized total-denoise performance claim. Full same-shape corresponding CPU/GPU step anchors used.'),indent=2)+'\n')
assert all(v['checks']['dispatch_pass'] for v in rows.values())
