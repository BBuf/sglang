"""Profile-shaped SANA-WM conv post-processing experiment, no model edits."""
import json
import os
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32

@triton.jit
def post(Y, X, Bias, N:tl.constexpr, C:tl.constexpr, S:tl.constexpr, GLU:tl.constexpr, HAS_BIAS:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    m=i<N
    c=(i//S)%C
    if GLU:
        base=(i//(C*S))*(2*C*S)+i%(C*S)
    else:
        base=i
    a=tl.load(X+base,m,0).to(tl.float32)
    if HAS_BIAS:
        a=round_bf16_to_fp32(a+tl.load(Bias+c,m,0).to(tl.float32))
    if GLU:
        g=tl.load(X+base+C*S,m,0).to(tl.float32)
        if HAS_BIAS:
            g=round_bf16_to_fp32(g+tl.load(Bias+c+C,m,0).to(tl.float32))
        g=round_bf16_to_fp32(g*tl.sigmoid(g))
        a=a*g
    else:
        a=a*tl.sigmoid(a)
    tl.store(Y+i,a,m)

def fused(x,bias=None,glu=False):
    b,c,h,w=x.shape
    if glu: c//=2
    y=torch.empty((b,c,h,w),device=x.device,dtype=x.dtype)
    post[(triton.cdiv(y.numel(),1024),)](y,x,bias,y.numel(),c,h*w,glu,bias is not None,1024)
    return y

def compare(a,b):
    return dict(exact=torch.equal(a,b),max_abs=(a-b).abs().max().item(),neq=(a!=b).sum().item(),numel=a.numel())

def ms(fn):
    return float(triton.testing.do_bench_cudagraph(fn,rep=1000))

def main():
    torch.manual_seed(42)
    report={'device':torch.cuda.get_device_name(),'torch':torch.__version__,'time':time.time(),'checks':[]}
    with torch.inference_mode():
        for seed in range(3):
            torch.manual_seed(seed)
            x=torch.randn((14,13440,22,40),device='cuda',dtype=torch.bfloat16)
            bias=torch.randn(13440,device='cuda',dtype=torch.bfloat16)
            for glu in (False,True):
                for has_bias in (False,True):
                    if not glu and not has_bias: continue
                    def ref():
                        z=x+bias[None,:,None,None] if has_bias else x
                        if glu:
                            a,g=z.chunk(2,1)
                            return a*F.silu(g)
                        return F.silu(z)
                    def opt(): return fused(x,bias if has_bias else None,glu)
                    row=dict(seed=seed,glu=glu,bias=has_bias,**compare(ref(),opt()))
                    if seed==0:
                        row.update(ref_ms=ms(ref),candidate_ms=ms(opt))
                    report['checks'].append(row)
                    print(json.dumps(row),flush=True)
            del x,bias
        x=torch.randn((14,2240,22,40),device='cuda',dtype=torch.bfloat16)
        conv=torch.nn.Conv2d(2240,13440,1,bias=True,device='cuda',dtype=torch.bfloat16)
        def native(): return F.silu(conv(x))
        def optconv(): return fused(F.conv2d(x,conv.weight,None),conv.bias)
        report['inverted_conv']=dict(**compare(native(),optconv()),native_ms=ms(native),candidate_ms=ms(optconv))
        print('inverted_conv',json.dumps(report['inverted_conv']),flush=True)
        x=native()
        dw=torch.nn.Conv2d(13440,13440,3,padding=1,groups=13440,bias=True,device='cuda',dtype=torch.bfloat16)
        actual=dw(x)
        split=F.conv2d(x,dw.weight,None,padding=1,groups=13440)+dw.bias[None,:,None,None]
        report['depthwise_bias_split']=compare(actual,split)
        print('depthwise_bias_split',json.dumps(report['depthwise_bias_split']),flush=True)
    with open(os.environ['SANA_BENCH_RESULT'],'w') as f:json.dump(report,f,indent=2)

if __name__=='__main__':main()
