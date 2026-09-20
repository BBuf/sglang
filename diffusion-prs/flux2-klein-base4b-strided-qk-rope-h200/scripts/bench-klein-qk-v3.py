"""Native packed Q/K chain vs the fused production site; no model BCG claim."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys

p=argparse.ArgumentParser();p.add_argument('--ncu',choices=['baseline','candidate']);args=p.parse_args()
root=Path('/campaign');repo=root/'candidate-klein-qk-pair-layout'
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(repo/'python'),FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0,str(repo/'python'))
import torch
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm,apply_qk_norm_with_optional_rope
from sglang.multimodal_gen.runtime.models.dits.flux_2 import _flux2_single_qk_rope,_FLUX2_STRIDED_QK_ROPE

def measure(fn):
 for _ in range(10):fn()
 torch.cuda.synchronize()
 samples=[]
 for _ in range(40):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
 return dict(median_us=statistics.median(samples),samples_us=samples)

with (root/'gpu.lock').open('a') as lock,torch.inference_mode():
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.manual_seed(42);records=[]
 for batch,tokens in ([(1,4608)] if args.ncu else [(1,512),(1,4608),(2,257)]):
  packed=torch.randn(batch,tokens,9*3072,device='cuda',dtype=torch.bfloat16)
  q,k,_=packed[:,:,:3*3072].chunk(3,dim=-1);q=q.unflatten(-1,(24,128));k=k.unflatten(-1,(24,128))
  qn,kn=(RMSNorm(128,eps=1e-6).to('cuda',torch.bfloat16) for _ in range(2))
  qn.weight.normal_();kn.weight.normal_()
  angles=torch.randn(tokens,64,device='cuda');cache=torch.cat([angles.cos(),angles.sin()],dim=-1)
  baseline=lambda:apply_qk_norm_with_optional_rope(q,k,qn,kn,128,cache,is_neox=False,allow_inplace=True,allow_strided_qk=False)
  candidate=lambda:_flux2_single_qk_rope(q,k,qn,kn,128,cache,None)
  expected=baseline();actual=candidate()
  assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,expected))
  assert _FLUX2_STRIDED_QK_ROPE.verified and not _FLUX2_STRIDED_QK_ROPE.disabled
  if args.ncu:
   fn=baseline if args.ncu=='baseline' else candidate
   for _ in range(10):fn()
   torch.cuda.synchronize();torch.cuda.nvtx.range_push('profile');fn();torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
  else:
   row=dict(batch=batch,tokens=tokens,heads=24,head_dim=128,packed_stride=packed.stride(),eager={})
   for name,fn in [('baseline',baseline),('candidate',candidate)]:row['eager'][name]=measure(fn)
   row['standalone_graph']={}
   for name,fn in [('baseline',baseline),('candidate',candidate)]:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    row['standalone_graph'][name]=measure(graph.replay)
   records.append(row)
   print(json.dumps({k:v for k,v in row.items() if k not in ('eager','standalone_graph')}|{mode:{arm:values['median_us'] for arm,values in row[mode].items()} for mode in ['eager','standalone_graph']}),flush=True)
 if not args.ncu:(root/'artifacts/flux2-klein-base-4b/qk-microbench-v3.json').write_text(json.dumps(records,indent=2))
