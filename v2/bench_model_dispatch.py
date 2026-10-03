import json,os,time,statistics
from pathlib import Path
import torch
from sglang.multimodal_gen.runtime.models.dits import glm_image, ernie_image
from sglang.srt.layers.layernorm import RMSNorm
from bench_cleanup import fingerprints

@torch.inference_mode()
def main():
    torch.manual_seed(1234)
    result=[]
    for seq in (17,4096):
        x=torch.randn(1,seq,4096,device='cuda',dtype=torch.bfloat16)
        row=torch.randn(1,4096,device=x.device,dtype=x.dtype)
        vec=row[:,None]
        q=torch.randn(1,seq,32,128,device=x.device,dtype=x.dtype)
        k=torch.randn_like(q)
        ln=torch.nn.LayerNorm(4096,elementwise_affine=False,eps=1e-6).cuda()
        qln=torch.nn.LayerNorm(128,elementwise_affine=False,eps=1e-6).cuda()
        rms=RMSNorm(4096,eps=1e-6).to(device=x.device,dtype=x.dtype)
        cases=[('glm_ln',lambda:glm_image._glm_ln_modulate(ln,x,row,row,x.dtype),glm_image._GLM_LN_MOD),('glm_qk_ln',lambda:glm_image._glm_qk_layernorm(qln,qln,q,k,q.dtype),glm_image._GLM_QK_LN),('ernie_rms',lambda:ernie_image._ernie_norm_scale_shift(rms,x,vec,vec),ernie_image._ERNIE_NORM)]
        for name,fn,gate in cases:
            output=fingerprints(fn())
            assert gate.verified and not gate.disabled,(name,gate.verified,gate.disabled)
            for _ in range(20):fn()
            samples=[]
            for _ in range(5):
                torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(1000):fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter()-start)*1000)
            result.append(dict(name=f'{name}/{seq}',eager_us=statistics.median(samples),samples_us=samples,output=output,verified=True))
    Path(os.environ['OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
