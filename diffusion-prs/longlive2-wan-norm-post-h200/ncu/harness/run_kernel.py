import argparse
import importlib
import json
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('--provider',choices=['baseline','fused'],required=True);p.add_argument('--channels-last',action='store_true');args=p.parse_args()
m=importlib.import_module('sglang.kernels.ops.diffusion.norm.wan_norm_silu_post')
load=m.load_jit
def load_with_lineinfo(*a,**kw):
    kw['extra_cuda_cflags']=kw.get('extra_cuda_cflags',[])+['-lineinfo']
    return load(*a,**kw)
m.load_jit=load_with_lineinfo
torch.manual_seed(1729)
with torch.inference_mode():
    x=torch.randn(1,256,4,240,416,device='cuda',dtype=torch.bfloat16)
    if args.channels_last:x=x.contiguous(memory_format=torch.channels_last_3d)
    gamma=torch.randn(256,1,1,1,device='cuda',dtype=torch.bfloat16)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        denominator=x.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12)
    def baseline():return F.silu((x/denominator)*16.0*gamma+0.0)
    def fused():return m.wan_norm_silu_post(x,denominator,gamma,scale=16.0)
    assert torch.equal(baseline().view(torch.int32),fused().view(torch.int32))
    fn=baseline if args.provider=='baseline' else fused
    for _ in range(10):y=fn()
    torch.cuda.synchronize()
    Path(__file__).with_name('metadata-'+args.provider+'.json').write_text(json.dumps(dict(source_head=subprocess.check_output(['git','-C','/campaign/candidate-longlive2-wan-norm','rev-parse','HEAD'],text=True).strip(),torch=torch.__version__,torch_git=torch.version.git_version,shape=list(x.shape),stride=x.stride(),dtype=str(x.dtype),denominator_dtype=str(denominator.dtype),gamma_dtype=str(gamma.dtype),extra_cuda_cflags=['--fmad=false','-lineinfo']),indent=2))
    torch.cuda.cudart().cudaProfilerStart()
    y=fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
