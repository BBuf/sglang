"""Replay observed native inputs against the existing fused RoPE kernel."""
import hashlib
import json
from pathlib import Path

import torch
import triton
from sglang.kernels.ops.diffusion.rope.rope_rotate_half_fp32 import (
    _rope_rotate_half_fp32_kernel,
    fused_rope_rotate_half_fp32,
)
from sglang.multimodal_gen.runtime.models.dits.anima import _anima_rope_eager

torch.set_num_threads(4)
root = Path('/task')
target = root / 'evidence/preflight/anima-rope-numerics1.json'
assert not target.exists()
path = next((root/'outputs/diagnostics/anima-tensor-audit1/sgl').glob('*/*rope*.pt'))
data = torch.load(path, weights_only=True)
q, k, cos, sin = [data[name].cuda() for name in ['q', 'k', 'cos', 'sin']]
with torch.inference_mode():
    eager = _anima_rope_eager(q, k, cos, sin)
    fused = fused_rope_rotate_half_fp32(q, k, cos, sin)
    float_out = [torch.empty_like(x, dtype=torch.float32) for x in [q,k]]
    _, seq, heads, dim = q.shape
    _rope_rotate_half_fp32_kernel[(triton.cdiv(q.numel(),1024),2)](
        q,k,cos,sin,*float_out,q.numel(),seq,heads,dim,BLOCK=1024,enable_fp_fusion=False)
    float_refs = []
    for x in [q,k]:
        x1,x2 = x.chunk(2,dim=-1)
        float_refs.append(x.float()*cos[None,:,None,:] + torch.cat((-x2,x1),-1).float()*sin[None,:,None,:])
    rows = []
    for label, x, ref, out, fr, fo, saved in zip(['q','k'],[q,k],eager,fused,float_refs,float_out,data['output']):
        mismatch = ref != out
        indices = mismatch.nonzero()[:16].tolist()
        examples = []
        for b,s,h,d in indices:
            d2 = d+64 if d<64 else d-64
            examples.append(dict(index=[b,s,h,d],x=float(x[b,s,h,d]),rotated_input=float(x[b,s,h,d2]),
                cos=float(cos[s,d]),sin=float(sin[s,d]),eager=float(ref[b,s,h,d]),fused=float(out[b,s,h,d]),
                float_reference=float(fr[b,s,h,d]),float_kernel=float(fo[b,s,h,d]),
                float_reference_bits=int(fr[b,s,h,d].view(torch.int32)),float_kernel_bits=int(fo[b,s,h,d].view(torch.int32))))
        rows.append(dict(name=label,eager_matches_observed=torch.equal(ref.cpu(),saved),
            mismatched=int(mismatch.sum()),max_abs=float((ref.float()-out.float()).abs().max()),
            float_precast_mismatched=int((fr!=fo).sum()),float_precast_max_abs=float((fr-fo).abs().max()),
            torch_cast_of_float_kernel_matches_eager=torch.equal(fo.to(ref.dtype),ref),
            triton_store_matches_torch_cast=torch.equal(out,fo.to(out.dtype)),examples=examples))
result = dict(scope='Existing kernel numerical replay only; no candidate or performance claim',
    device=torch.cuda.get_device_name(),torch=str(torch.__version__),triton=triton.__version__,
    input_file=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
