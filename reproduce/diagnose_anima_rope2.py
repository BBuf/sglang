"""Validate opaque FP32 multiply boundaries on the observed RoPE fallback."""
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import triton
import anima_rope_mul_variant as candidate
from sglang.kernels.ops.diffusion.rope import rope_rotate_half_fp32 as original
from sglang.multimodal_gen.runtime.models.dits.anima import _anima_rope_eager

torch.set_num_threads(4)
root=Path('/task')
out=root/'outputs/diagnostics/anima-rope-numerics2'
out.mkdir(parents=True,exist_ok=False)
path=next((root/'outputs/diagnostics/anima-tensor-audit1/sgl').glob('*/*rope*.pt'))
data=torch.load(path,weights_only=True)
actual=[data[name].cuda() for name in ['q','k','cos','sin']]
generator=torch.Generator(device='cuda').manual_seed(123)
cases=[('observed_warmup',actual)]
for dtype in [torch.bfloat16,torch.float16]:
    for shape in [(1,17,3,128),(2,127,4,128),(1,4096,16,128)]:
        q,k=[torch.randn(shape,dtype=dtype,device='cuda',generator=generator) for _ in range(2)]
        angles=torch.randn(shape[1],shape[-1],device='cuda',generator=generator)
        cases.append((str(dtype)+str(shape),[q,k,angles.cos(),angles.sin()]))
rows=[]
with torch.inference_mode():
    for name,args in cases:
        ref=_anima_rope_eager(*args)
        baseline=original.fused_rope_rotate_half_fp32(*args)
        new=candidate.fused_rope_rotate_half_fp32(*args)
        rows.append(dict(name=name,baseline_mismatches=[int((x!=y).sum()) for x,y in zip(baseline,ref)],
            candidate_mismatches=[int((x!=y).sum()) for x,y in zip(new,ref)],candidate_exact=all(torch.equal(x,y) for x,y in zip(new,ref))))
    q,k,cos,sin=actual
    for label,module in [('original',original),('opaque_mul',candidate)]:
        outputs=[torch.empty_like(q),torch.empty_like(k)]
        _,seq,heads,dim=q.shape
        compiled=module._rope_rotate_half_fp32_kernel[(triton.cdiv(q.numel(),1024),2)](
            q,k,cos,sin,*outputs,q.numel(),seq,heads,dim,BLOCK=1024,enable_fp_fusion=False)
        (out/(label+'.ptx')).write_text(compiled.asm['ptx'])
        (out/(label+'.llir')).write_text(compiled.asm['llir'])
        (out/(label+'.cubin')).write_bytes(compiled.asm['cubin'])
        sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out/(label+'.cubin'))],text=True)
        (out/(label+'.sass')).write_text(sass)
result=dict(scope='Diagnostic variant, no source mutation or performance claim',torch=str(torch.__version__),
    triton=triton.__version__,device=torch.cuda.get_device_name(),rows=rows,
    all_candidate_exact=all(x['candidate_exact'] for x in rows),
    codegen={label:dict(ffma_count=(out/(label+'.sass')).read_text().count(' FFMA '),
        packed_mul_count=(out/(label+'.ptx')).read_text().count('mul.rn.f32x2'),
        scalar_mul_count=(out/(label+'.ptx')).read_text().count('mul.rn.f32 ')) for label in ['original','opaque_mul']},
    artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    variant_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
target=root/'evidence/preflight/anima-rope-numerics2.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
assert result['all_candidate_exact']
