"""CPU contract checks for the gfx950 native MXFP8 MoE entry point.

The GPU numerics live in test_minimax_m3_mxfp8.py; these run before any launch.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe import mxfp8_moe_amd_gfx95
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fused_experts_mxfp8(**kwargs):
    t = torch.empty(0)
    return mxfp8_moe_amd_gfx95.fused_experts_mxfp8(
        t, t, t, t, t, t, t, gate_up_interleaved=False, **kwargs
    )


class TestFusedExpertsMxfp8Contract(CustomTestCase):
    def test_rejects_configs_outside_minimax_m3(self):
        """Plain SiLU (no gemm1_alpha) and swiglu_limit must raise instead of
        running as SwiGLU-OAI with a made-up alpha or a dropped clamp."""
        for kwargs in (
            dict(gemm1_alpha=None),
            dict(gemm1_alpha=1.702, swiglu_limit=10.0),
        ):
            with (
                self.subTest(**kwargs),
                patch.object(mxfp8_moe_amd_gfx95, "fused_moe_mxfp8_native") as native,
            ):
                with self.assertRaisesRegex(NotImplementedError, "MiniMax-M3"):
                    _fused_experts_mxfp8(**kwargs)
                native.assert_not_called()

    def test_minimax_m3_config_reaches_kernel(self):
        with patch.object(mxfp8_moe_amd_gfx95, "fused_moe_mxfp8_native") as native:
            _fused_experts_mxfp8(gemm1_alpha=1.5, gemm1_limit=7.0)
        kwargs = native.call_args.kwargs
        self.assertEqual(
            (kwargs["alpha"], kwargs["beta"], kwargs["limit"]), (1.5, 1.0, 7.0)
        )


if __name__ == "__main__":
    unittest.main()
