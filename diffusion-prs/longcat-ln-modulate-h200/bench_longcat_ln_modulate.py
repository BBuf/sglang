"""Profile-derived LongCat LayerNorm/modulation feasibility check."""



import torch
import torch.nn.functional as F
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_layernorm_modulate


def reference(x, scale, shift):
    return F.layer_norm(x, (3072,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]


def fused(x, scale, shift):
    return fused_layernorm_modulate(x, scale, shift, 1e-6)


@marker.parametrize('seq', [512, 4096, 4608], [17])
@marker.benchmark('impl', ['eager', 'fused'], unit='us')
def benchmark(seq, impl):
    torch.manual_seed(42)
    x = torch.randn(1, seq, 3072, device='cuda', dtype=torch.bfloat16)
    modulation = torch.randn(1, 6 * 3072, device='cuda', dtype=torch.bfloat16)
    shift, scale, *_ = modulation.chunk(6, dim=-1)
    assert torch.equal(reference(x, scale, shift), fused(x, scale, shift))
    return marker.do_bench(reference if impl == 'eager' else fused, input_args=(x, scale, shift))


if __name__ == '__main__':
    benchmark.run()
