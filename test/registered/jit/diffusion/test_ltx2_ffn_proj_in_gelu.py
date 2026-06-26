import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
    ltx2_bias_gelu_tanh_inplace,
)
from sglang.multimodal_gen.runtime.models.dits import ltx_2 as ltx2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16
ATOL = 4e-2
RTOL = 2e-2


class UnquantizedLinearMethod:
    pass


class FakeProjIn(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=DEVICE, dtype=DTYPE)
        )
        self.bias = torch.nn.Parameter(
            torch.empty(out_features, device=DEVICE, dtype=DTYPE)
        )
        self.gather_output = False
        self.skip_bias_add = False
        self.quant_method = UnquantizedLinearMethod()


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    ltx2._LTX2_FUSED_FFN_PROJ_IN_GELU = None
    ltx2._LTX2_FUSED_FFN_PROJ_IN_GELU_UNAVAILABLE = False
    ltx2._LTX2_FUSED_FFN_PROJ_IN_GELU_RUNTIME_DISABLED = False


@torch.no_grad()
def test_ltx2_bias_gelu_tanh_inplace_matches_pytorch():
    x = torch.randn(3, 11, 257, device=DEVICE, dtype=DTYPE) * 0.25
    bias = torch.randn(257, device=DEVICE, dtype=DTYPE) * 0.25

    actual = ltx2_bias_gelu_tanh_inplace(x.clone(), bias)
    expected = F.gelu(x + bias, approximate="tanh")

    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@torch.no_grad()
def test_ltx2_fused_ffn_proj_in_gelu_matches_reference():
    x = torch.randn(1, 17, 128, device=DEVICE, dtype=DTYPE) * 0.25
    proj_in = FakeProjIn(128, 512)
    proj_in.weight.copy_(torch.randn_like(proj_in.weight) * 0.05)
    proj_in.bias.copy_(torch.randn_like(proj_in.bias) * 0.25)

    actual = ltx2._ltx2_try_fused_ffn_proj_in_gelu(x, proj_in)
    assert actual is not None

    expected = F.gelu(F.linear(x, proj_in.weight, proj_in.bias), approximate="tanh")
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_ltx2_fused_ffn_proj_in_gelu_ignores_grad_enabled():
    x = torch.randn(1, 4, 128, device=DEVICE, dtype=DTYPE)
    proj_in = FakeProjIn(128, 512)

    actual = ltx2._ltx2_try_fused_ffn_proj_in_gelu(x, proj_in)
    assert actual is None
