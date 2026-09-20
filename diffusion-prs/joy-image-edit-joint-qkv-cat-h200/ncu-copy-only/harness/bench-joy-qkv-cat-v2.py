"""Native three cats vs the production joint-copy helper on profile shapes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys

p=argparse.ArgumentParser();p.add_argument('--ncu',choices=['baseline','candidate']);args=p.parse_args()
root=Path('/campaign');repo=root/'candidate-joy-qkv-cat-v2'
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(repo/'python'),FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0,str(repo/'python'))
import torch
from sglang.multimodal_gen.runtime.models.dits.joy_image import _joy_joint_qkv,_JOY_QKV_CAT

def measure(fn):
 for _ in range(10):fn()
 torch.cuda.synchronize();samples=[]
 for _ in range(40):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
 return dict(median_us=statistics.median(samples),samples_us=samples)

with (root/'gpu.lock').open('a') as lock,torch.inference_mode():
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.manual_seed(42);records=[]
 for batch,image_tokens,text_tokens,heads in ([(1,8048,1004,32)] if args.ncu else [(1,8048,1004,32),(1,4096,512,32),(1,2048,256,32),(2,257,13,8)]):
  inputs=[]
  for tokens in (image_tokens,text_tokens):
   packed=torch.randn(batch,tokens,3,heads,128,device='cuda',dtype=torch.bfloat16)
   q,k,v=packed.unbind(2);inputs.extend([q.contiguous(),k.contiguous(),v])
  baseline=lambda:tuple(torch.cat((inputs[i],inputs[i+3]),dim=1) for i in range(3))
  candidate=lambda:_joy_joint_qkv(*inputs)
  expected=baseline();actual=candidate()
  assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,expected))
  assert _JOY_QKV_CAT.verified and not _JOY_QKV_CAT.disabled
  if args.ncu:
   fn=baseline if args.ncu=='baseline' else candidate
   for _ in range(10):fn()
   torch.cuda.synchronize();torch.cuda.nvtx.range_push('profile');fn();torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
  else:
   row=dict(batch=batch,image_tokens=image_tokens,text_tokens=text_tokens,heads=heads,dim=128,input_strides=[x.stride() for x in inputs],eager={})
   for name,fn in [('baseline',baseline),('candidate',candidate)]:row['eager'][name]=measure(fn)
   row['standalone_graph']={}
   for name,fn in [('baseline',baseline),('candidate',candidate)]:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    row['standalone_graph'][name]=measure(graph.replay)
   records.append(row)
   print(json.dumps({k:v for k,v in row.items() if k not in ('eager','standalone_graph')}|{mode:{arm:values['median_us'] for arm,values in row[mode].items()} for mode in ['eager','standalone_graph']}),flush=True)
 if not args.ncu:(root/'artifacts/joy-image-edit/qkv-cat-microbench-v2.json').write_text(json.dumps(records,indent=2))
