"""Benchmark the living PixelUMM reference and one-pass ordered reductions on actual row shapes."""
import json
import os
from pathlib import Path
import statistics

import torch
import triton
from sglang.kernels.ops.diffusion.norm.rmsnorm_cast_before_weight_bitexact import _ordered_norm_kernel
from sglang.multimodal_gen.runtime.models.vaes.nomas import _compiled_rms_scale, rmsnorm_mean_square_bitexact

assert torch.cuda.device_count()==1
assert torch.cuda.get_device_capability()==(12,0)
p=Path('decoder-operator-cuda-events-local.json')
rows=[]
def save(status):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=status,device=torch.cuda.get_device_name(),torch=torch.__version__,
        comparison='Unmodified production mean-square followed by the existing compiled precision-preserving scale versus the same source ordered RMS kernel at width 4096',rows=rows),indent=2)+'\n');tmp.replace(p)
def measure(fn):
    for _ in range(3):out=fn()
    events=[torch.cuda.Event(enable_timing=True) for _ in range(8)];events[0].record()
    for event in events[1:]:
        for _ in range(4):out=fn()
        event.record()
    events[-1].synchronize()
    return out,[a.elapsed_time(b)/4 for a,b in zip(events,events[1:])]
torch.manual_seed(20261127)
with torch.inference_mode():
    for size in (16,17,1001,7040,21120,35200):
        for amplitude in (False,True):
            x=torch.randn(1,size,4096,device='cuda',dtype=torch.bfloat16)
            if amplitude:
                x.mul_(torch.exp2(torch.randint(-12,12,x.shape,device='cuda',dtype=torch.int8).float()).bfloat16())
            weight=torch.randn(4096,device='cuda',dtype=x.dtype)
            reference=lambda:_compiled_rms_scale()(x,rmsnorm_mean_square_bitexact(x),1e-6,weight,x.dtype)
            result,baseline=measure(reference)
            row=dict(shape=list(x.shape),varying_exponents=amplitude,reference_ms=baseline,variants=[])
            for nr,warps in ((1,4),(2,4),(4,4),(8,4),(8,8),(16,8)):
                output=torch.empty_like(x)
                def launch():
                    _ordered_norm_kernel[(triton.cdiv(size,nr),)](x,weight,output,size,4096,nr,4096,1e-6,num_warps=warps,enable_fp_fusion=False)
                    return output
                try:
                    actual,times=measure(launch)
                    different=int(torch.count_nonzero(result.view(torch.int16)!=actual.view(torch.int16)))
                    row['variants'].append(dict(rows_per_cta=nr,warps=warps,ms=times,changed_bits=different))
                except Exception as exc:row['variants'].append(dict(rows_per_cta=nr,warps=warps,error=repr(exc)))
            _,last=measure(reference);row['reference_repeat_ms']=last
            rows.append(row);save('running')
save('terminal')
