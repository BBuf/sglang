"""Native LingBot BF16 input/output with FP32 LN at the profiled shape."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from sglang.kernels.ops.diffusion import try_fused_fp32_layernorm_bf16

p=argparse.ArgumentParser()
p.add_argument('--provider',choices=['baseline','fused'],required=True)
p.add_argument('--site',choices=['norm1','cross_norm'],required=True)
a=p.parse_args()
torch.manual_seed(1729)
with torch.inference_mode():
    x=torch.randn(1,4680,5120,device='cuda',dtype=torch.bfloat16)
    scale=torch.randn(1,5120,device='cuda',dtype=torch.float32)
    shift=torch.randn_like(scale)
    affine=a.site=='cross_norm'
    def baseline():
        if affine:return F.layer_norm(x.float(),(5120,),scale[0],shift[0],1e-6).bfloat16()
        return (F.layer_norm(x.float(),(5120,),eps=1e-6)*(1+scale[:,None])+shift[:,None]).bfloat16()
    def fused():return try_fused_fp32_layernorm_bf16(x,scale,shift,1e-6,affine=affine)
    assert torch.equal(baseline().view(torch.int16),fused().view(torch.int16))
    fn=baseline if a.provider=='baseline' else fused
    for _ in range(10):y=fn()
    torch.cuda.synchronize()
    repo=Path('/campaign/candidate-lingbot-current')
    metadata=dict(source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),torch=torch.__version__,torch_git=torch.version.git_version,shape=list(x.shape),stride=x.stride(),dtype=str(x.dtype),affine_dtype=str(scale.dtype),site=a.site,provider=a.provider,eps=1e-6,implicit_fp_fusion=False,explicit_affine_fma=affine)
    Path(__file__).with_name('metadata-'+a.site+'-'+a.provider+'.json').write_text(json.dumps(metadata,indent=2))
    torch.cuda.cudart().cudaProfilerStart()
    y=fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
