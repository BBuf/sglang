import os
import bench_cold_eager as bench
from bench_workloads import rand
import torch

def workloads():
 from sglang.multimodal_gen.runtime.models.dits import ernie_image, ideogram
 from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import apply_rotary_pos_emb
 from sglang.srt.layers.layernorm import RMSNorm
 for seq in (17,4096):
  x=rand((1,seq,4096)); row=rand((1,1,4096)); norm=RMSNorm(4096,eps=1e-6).to(device=x.device,dtype=x.dtype)
  def ernie_gated(x,u,g,s,t):return ernie_image._ernie_gated_norm_scale_shift(norm,x,u,g,s,t)
  yield f'model_ernie_gated/{seq}',ernie_gated,(x,rand(x.shape),row,row,row),None
  assert ernie_image._ERNIE_GATED_NORM.verified and not ernie_image._ERNIE_GATED_NORM.disabled
  q,k=rand((1,seq,32,128)),rand((1,seq,32,128))
  cos,sin=rand((seq,128)),rand((seq,128))
  yield f'model_ernie_rope/{seq}',ernie_image._ernie_rope,(q,cos,sin),None
  assert ernie_image._ERNIE_ROPE.verified and not ernie_image._ERNIE_ROPE.disabled
  yield f'model_ideogram_rope/{seq}',ideogram._ideogram_rope,(q,k,cos.view(1,seq,1,128),sin.view(1,seq,1,128)),None
  assert ideogram._IDEOGRAM_ROPE.verified and not ideogram._IDEOGRAM_ROPE.disabled
  if seq>=128:
   q,k=rand((1,32,seq,64)),rand((1,32,seq,64))
   yield f'model_sensenova_rope/{seq}',apply_rotary_pos_emb,(q,k,rand((1,seq,64)),rand((1,seq,64))),None
bench.workloads=workloads
bench.extra_workloads=lambda:iter(())
bench.main()
