"""Exercise production VAE fallback and autograd bypass; no timing evidence."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.models.vaes import nomas

out=Path(sys.argv[1]);assert not out.exists()
torch.manual_seed(20261115)
rows=[]
with torch.inference_mode():
    inputs=[
        torch.randn(17,67,device='cuda',dtype=torch.float32),
        torch.randn(15,4096,device='cuda',dtype=torch.bfloat16),
        torch.randn(17,8192,device='cuda',dtype=torch.bfloat16)[:,::2],
        torch.randn(2,19,8,128,device='cuda',dtype=torch.bfloat16)[:,1:-1,:,64:],
    ]
    for x in inputs:
        if hasattr(nomas,'can_use_rmsnorm_mean_square'):
            assert not nomas.can_use_rmsnorm_mean_square(x)
        actual=nomas._mean_square(x)
        assert torch.equal(actual,x.float().square().mean(-1,keepdim=True))
        rows.append(dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),exact=True,
                         output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()))
for device,dtype in [('cpu',torch.float32),('cuda',torch.bfloat16)]:
    x=torch.randn(17,4096,device=device,dtype=dtype,requires_grad=True)
    layer=nomas.RMSNorm(4096,1e-6).to(device=device,dtype=dtype)
    xf=x.float()
    reference=layer.weight*(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)).to(dtype)
    actual=layer(x)
    assert torch.equal(actual,reference)
    left=torch.autograd.grad(actual.float().sum(),[x,layer.weight],retain_graph=True)
    right=torch.autograd.grad(reference.float().sum(),[x,layer.weight])
    assert all(torch.equal(a,b) for a,b in zip(left,right))
    rows.append(dict(device=device,dtype=str(dtype),autograd_output_and_gradients_exact=True))
out.write_text(json.dumps(dict(benchmark=False,checks=rows),indent=2)+'\n')
print(json.dumps(dict(checks=len(rows),passed=True)))
