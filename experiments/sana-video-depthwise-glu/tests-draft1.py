"""Unapplied cases for the existing test_model_fast_paths.py fixture.

No tests have run from this draft. Preserve final CI registration and imports
when integrating it after the native-model admission.
"""

import pytest
import torch
import torch.nn.functional as F


_SM120 = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() == (12, 0)
)


@pytest.mark.skipif(
    not _SM120, reason="SANA-Video depthwise fast path is enabled on SM120"
)
@pytest.mark.parametrize("shape", [(2, 34, 1, 1), (2, 192, 7, 11), (21, 13440, 30, 52)])
@torch.no_grad()
def test_sana_video_depthwise_glu_rounding_and_row_tails(shape):
    from sglang.kernels.ops.diffusion import fused_depthwise_conv3x3_bias_glu

    torch.manual_seed(606)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    weight = torch.randn((shape[1], 1, 3, 3), device=x.device, dtype=x.dtype) * 0.1
    bias = torch.randn(shape[1], device=x.device, dtype=x.dtype) * 0.1
    raw = F.conv2d(x, weight, None, padding=1, groups=shape[1])
    hidden, gate = (raw + bias[None, :, None, None]).chunk(2, dim=1)
    actual = fused_depthwise_conv3x3_bias_glu(x, weight, bias)
    assert torch.equal(actual, hidden * F.silu(gate))
    assert actual.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(
    not _SM120, reason="SANA-Video depthwise fast path is enabled on SM120"
)
@torch.no_grad()
def test_sana_video_depthwise_gate_capture_and_changed_inputs(monkeypatch):
    from sglang.kernels.ops.diffusion import BitExactFusionGate
    from sglang.multimodal_gen.runtime.models.dits import sana_video
    from sglang.multimodal_gen.runtime.models.dits.sana import sana_conv_bias_glu

    gate = BitExactFusionGate("test depthwise", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    torch.manual_seed(606)
    module = sana_video.GLUMBTempConv(32, 3).cuda().bfloat16()
    module.conv_depth = (
        torch.nn.Conv2d(13440, 13440, 3, padding=1, groups=13440).cuda().bfloat16()
    )
    x = torch.randn((2, 13440, 30, 52), device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    expected = sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unverified = module._depthwise_glu(x)
    graph.replay()
    assert torch.equal(unverified, expected)
    assert not gate.verified and not gate.disabled
    assert torch.equal(module._depthwise_glu(x), expected)
    assert gate.verified and not gate.disabled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = module._depthwise_glu(x)
    x.add_(0.125)
    module.conv_depth.bias.mul_(0.75)
    graph.replay()
    assert torch.equal(
        actual, sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    )
    gate = BitExactFusionGate("test mismatch", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    expected = sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    with monkeypatch.context() as context:
        context.setattr(
            sana_video,
            "fused_depthwise_conv3x3_bias_glu",
            lambda *args: torch.zeros_like(expected),
        )
        assert torch.equal(module._depthwise_glu(x), expected)
    assert gate.disabled
    x.mul_(0.5)
    assert torch.equal(
        module._depthwise_glu(x),
        sana_conv_bias_glu(module.conv_depth, x, allow_eager=True),
    )


@pytest.mark.skipif(
    not _SM120, reason="SANA-Video depthwise fast path is enabled on SM120"
)
def test_sana_video_depthwise_grad_fallback(monkeypatch):
    from sglang.kernels.ops.diffusion import BitExactFusionGate
    from sglang.multimodal_gen.runtime.models.dits import sana_video
    from sglang.multimodal_gen.runtime.models.dits.sana import sana_conv_bias_glu

    gate = BitExactFusionGate("test autograd", per_signature=True)
    monkeypatch.setattr(sana_video, "_SANA_VIDEO_DEPTHWISE_GLU", gate)
    torch.manual_seed(606)
    module = sana_video.GLUMBTempConv(32, 3).cuda().bfloat16()
    module.conv_depth = (
        torch.nn.Conv2d(13440, 13440, 3, padding=1, groups=13440).cuda().bfloat16()
    )
    x = (
        torch.randn((2, 13440, 30, 52), device="cuda", dtype=torch.bfloat16)
        .to(memory_format=torch.channels_last)
        .requires_grad_()
    )
    actual = module._depthwise_glu(x)
    expected = sana_conv_bias_glu(module.conv_depth, x, allow_eager=True)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert not gate.verified and not gate.disabled
