"""Check existing rounded QK/RoPE arithmetic against native packed Klein Q/K."""
import fcntl
import json
import os
from pathlib import Path
import sys

root=Path('/campaign');repo=root/'candidate-klein-qk-prototype'
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(repo/'python'),FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0,str(repo/'python'))
import torch
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm_rope
from sglang.kernels.ops.diffusion.rope.flux2_qknorm_rope_triton import flux2_strided_qknorm_rope

with (root/'gpu.lock').open('a') as lock,torch.inference_mode():
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.manual_seed(42)
 records=[]
 for batch,tokens in [(1,4608),(2,257)]:
  normq=RMSNorm(128,eps=1e-6).to('cuda',torch.bfloat16)
  normk=RMSNorm(128,eps=1e-6).to('cuda',torch.bfloat16)
  normq.weight.normal_();normk.weight.normal_()
  angles=torch.randn(tokens,64,device='cuda')
  cache=torch.cat((angles.cos(),angles.sin()),dim=-1)
  for magnitude in [.005,1.,200.]:
   packed=torch.randn(batch,tokens,9*3072,device='cuda',dtype=torch.bfloat16)*magnitude
   q,k,v=packed[:,:,:3*3072].chunk(3,dim=-1)
   q=q.unflatten(-1,(24,128));k=k.unflatten(-1,(24,128))
   ref=apply_qk_norm_rope(q,k,normq,normk,128,cache,allow_strided_qk=False)
   for rounded in [True]:
    copy=packed.clone();cq,ck,_=copy[:,:,:3*3072].chunk(3,dim=-1)
    out=flux2_strided_qknorm_rope(cq.unflatten(-1,(24,128)),ck.unflatten(-1,(24,128)),normq.weight,normk.weight,cache,1e-6)
    row=dict(batch=batch,tokens=tokens,magnitude=magnitude,rounded=rounded,
     mismatches=[int((a.view(torch.int16)!=b.view(torch.int16)).sum()) for a,b in zip(out,ref)],
     max_error=[float((a.float()-b.float()).abs().max()) for a,b in zip(out,ref)],
     value_mlp_unchanged=torch.equal(copy[:,:,2*3072:],packed[:,:,2*3072:]))
    records.append(row);print(json.dumps(row),flush=True)
 (root/'artifacts/flux2-klein-base-4b/qk-native-triton-probe.json').write_text(json.dumps(records,indent=2))
