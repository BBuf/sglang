"""One exact, warmed Qwen Edit QKV epilogue launch for Nsight Compute."""
import torch
from sglang.kernels.ops.diffusion import try_fused_qwen_qkv_epilogue
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm_with_optional_rope

torch.manual_seed(42)
assert torch.cuda.get_device_capability() == (9, 0)
with torch.inference_mode():
    ni, nt, heads, dim = 8152, 1365, 24, 128
    img = [torch.randn(1, ni, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    txt = [torch.randn(1, nt, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    norms = [RMSNorm(dim, eps=1e-6).to(device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    for n in norms:n.weight.copy_(torch.randn_like(n.weight))
    def cache(n):
        a = torch.randn(n, dim//2, device='cuda')
        return torch.cat([a.cos(),a.sin()],dim=-1)
    ic, tc = cache(ni), cache(nt)
    ri,rt=[[t.clone() for t in tensors] for tensors in (img,txt)]
    iq,ik=apply_qk_norm_with_optional_rope(ri[0],ri[1],norms[0],norms[1],dim,ic,is_neox=False)
    tq,tk=apply_qk_norm_with_optional_rope(rt[0],rt[1],norms[2],norms[3],dim,tc,is_neox=False)
    expected=[torch.cat([tq,iq],1),torch.cat([tk,ik],1),torch.cat([rt[2],ri[2]],1)]
    def candidate():
        result=try_fused_qwen_qkv_epilogue(*img,*txt,*[n.weight for n in norms],ic,tc,1e-6,1e-6)
        assert result is not None
        return result
    actual=candidate()
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    for _ in range(5):actual=candidate()
    torch.cuda.synchronize()
    print('EXACT_AND_WARMED shape=8152/1365 heads=24 dim=128 bf16',flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    actual=candidate()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print('CAPTURE_COMPLETE',flush=True)
