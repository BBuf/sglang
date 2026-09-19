import json
from pathlib import Path
import torch
from sglang.kernels.ops.diffusion import fused_gelu_tanh_cat

torch.manual_seed(2026)
x=torch.randn(1,9233,12288,device="cuda",dtype=torch.bfloat16)
a=torch.randn(1,9233,3072,device="cuda",dtype=torch.bfloat16)
for _ in range(10): y=fused_gelu_tanh_cat(a,x)
torch.cuda.synchronize()
Path(__file__).with_name("metadata.json").write_text(json.dumps(dict(torch=torch.__version__,torch_git=torch.version.git_version,mlp_shape=list(x.shape),attn_shape=list(a.shape),dtype=str(x.dtype)),indent=2))
torch.cuda.cudart().cudaProfilerStart()
y=fused_gelu_tanh_cat(a,x)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
