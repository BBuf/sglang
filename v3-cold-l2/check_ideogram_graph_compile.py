import json,os
from pathlib import Path
import torch
from sglang.multimodal_gen.runtime.models.dits import ideogram
from bench_workloads import fingerprints

@torch.inference_mode()
def main():
 torch.manual_seed(731)
 rows=[]
 for b,s,h,d in ((1,17,32,128),(2,257,16,128)):
  args=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for shape in ((b,s,h,d),(b,s,h,d),(b,s,1,d),(b,s,1,d))]
  fn=ideogram._ideogram_rope
  def check(out,values):
   ref=ideogram.qwen3_apply_rotary_pos_emb(*values)
   assert all(torch.equal(a,c) for a,c in zip(out,ref))
  check(fn(*args),args)
  assert ideogram._IDEOGRAM_ROPE.verified and not ideogram._IDEOGRAM_ROPE.disabled
  compiled=torch.compile(fn,fullgraph=True)
  check(compiled(*args),args)
  torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): out=fn(*args)
  graph.replay();torch.cuda.synchronize();check(out,args)
  fresh=[torch.randn_like(x) for x in args]
  for a,c in zip(args,fresh):a.copy_(c)
  graph.replay();torch.cuda.synchronize();check(out,fresh)
  check(compiled(*args),fresh)
  rows.append(dict(shape=[b,s,h,d],compile_fullgraph=True,graph_replay_new_inputs=True,output=fingerprints(out)))
 Path(os.environ['OUTPUT']).write_text(json.dumps(rows,indent=2)+'\n')
 print('PASS: Ideogram fullgraph compile and graph replay with changed inputs',flush=True)
main()
