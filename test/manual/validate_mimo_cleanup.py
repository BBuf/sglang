import ast
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AttentionInterface, Qwen2Config
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from sglang.kernels.ops.attention.mimo_local_attention import mimo_local_attention
from sglang.srt.models.mimo_audio import _mimo_local_attention_forward, _register_mimo_local_attention

assert torch.cuda.get_device_capability()[0] == 9, 'Hopper validation must run on Hopper'

OLD_FORWARD = 'def _mimo_local_attention_forward(\n    module,\n    query,\n    key,\n    value,\n    attention_mask,\n    dropout=0.0,\n    scaling=None,\n    is_causal=None,\n    position_bias=None,\n    **kwargs,\n):\n    from transformers.integrations.sdpa_attention import sdpa_attention_forward\n\n    if (\n        query.is_cuda\n        and query.device == key.device == value.device\n        and query.dtype == key.dtype == value.dtype\n        and query.dtype in (torch.float16, torch.bfloat16)\n        and query.ndim == 4\n        and query.shape == key.shape == value.shape\n        and query.shape[0] > 0\n        and 0 < query.shape[2] <= 4\n        and query.shape[3] == 16\n        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1\n        and attention_mask is None\n        and position_bias is None\n        and dropout == 0.0\n        and not kwargs.get("output_attentions", False)\n        and torch.cuda.get_device_capability(query.device)[0] == 9\n    ):\n        from sglang.kernels.ops.attention.mimo_local_attention import (\n            mimo_local_attention,\n        )\n\n        causal = (\n            is_causal if is_causal is not None else getattr(module, "is_causal", True)\n        )\n        return mimo_local_attention(\n            query,\n            key,\n            value,\n            scale=scaling if scaling is not None else 0.25,\n            is_causal=bool(causal),\n        ), None\n    return sdpa_attention_forward(\n        module,\n        query,\n        key,\n        value,\n        attention_mask,\n        dropout=dropout,\n        scaling=scaling,\n        is_causal=is_causal,\n        position_bias=position_bias,\n        **kwargs,\n    )'
ARCHIVED_TEST = '"""Small audio attention, SDPA fallback and Transformers integration."""\n\nimport copy\nimport unittest\nfrom types import SimpleNamespace\nfrom unittest.mock import patch\n\nimport torch\nimport torch.nn.functional as F\nfrom torch.nn.attention import SDPBackend, sdpa_kernel\n\nfrom sglang.kernels.ops.attention.mimo_local_attention import mimo_local_attention\nfrom sglang.srt.models.mimo_audio import (\n    _mimo_local_attention_forward,\n    _register_mimo_local_attention,\n)\nfrom sglang.test.ci.ci_register import register_cuda_ci\nfrom sglang.test.test_utils import CustomTestCase\n\nregister_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")\n\n\n@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")\nclass TestMiMoLocalAttention(CustomTestCase):\n    def test_shapes_strides_causality_and_graph(self):\n        for dtype in [torch.bfloat16, torch.float16]:\n            for batch, heads, tokens in [\n                (3, 7, 1),\n                (3, 7, 2),\n                (3, 7, 3),\n                (32, 64, 4),\n                (1876, 64, 4),\n            ]:\n                for causal in [False, True]:\n                    for token_major in [False, True]:\n                        with self.subTest(\n                            dtype=dtype,\n                            shape=(batch, heads, tokens),\n                            causal=causal,\n                            token_major=token_major,\n                        ):\n                            if token_major:\n                                q, k, v = [\n                                    torch.randn(\n                                        batch,\n                                        tokens,\n                                        heads,\n                                        16,\n                                        device="cuda",\n                                        dtype=dtype,\n                                    ).transpose(1, 2)\n                                    for _ in range(3)\n                                ]\n                            else:\n                                q, k, v = [\n                                    torch.randn(\n                                        batch,\n                                        heads,\n                                        tokens,\n                                        16,\n                                        device="cuda",\n                                        dtype=dtype,\n                                    )\n                                    for _ in range(3)\n                                ]\n                            mimo_local_attention(q, k, v, is_causal=causal)\n                            graph = torch.cuda.CUDAGraph()\n                            with torch.cuda.graph(graph):\n                                actual = mimo_local_attention(q, k, v, is_causal=causal)\n                            for factor in [1.0, -1.0, 0.0]:\n                                q.mul_(factor)\n                                k.neg_()\n                                graph.replay()\n                                scores = q.float() @ k.float().transpose(-1, -2) * 0.25\n                                if causal:\n                                    allowed = torch.ones(\n                                        tokens, tokens, device="cuda", dtype=torch.bool\n                                    ).tril()\n                                    scores.masked_fill_(~allowed, -float("inf"))\n                                probabilities = torch.exp(\n                                    scores - scores.amax(dim=-1, keepdim=True)\n                                )\n                                denominator = probabilities.sum(dim=-1, keepdim=True)\n                                expected = (\n                                    (\n                                        (probabilities.to(dtype).float() @ v.float())\n                                        / denominator\n                                    )\n                                    .transpose(1, 2)\n                                    .to(dtype)\n                                )\n                                torch.testing.assert_close(\n                                    actual, expected, rtol=0.02, atol=0.02\n                                )\n                                error = (\n                                    (actual.float() - expected.float())\n                                    .square()\n                                    .mean()\n                                    .sqrt()\n                                )\n                                reference = (\n                                    expected.float()\n                                    .square()\n                                    .mean()\n                                    .sqrt()\n                                    .clamp_min(1e-6)\n                                )\n                                self.assertLess(float(error / reference), 0.001)\n                                self.assertTrue(actual.is_contiguous())\n                                if (\n                                    dtype == torch.bfloat16\n                                    and heads == 64\n                                    and torch.cuda.get_device_capability()[0] == 9\n                                ):\n                                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):\n                                        cudnn = F.scaled_dot_product_attention(\n                                            q, k, v, is_causal=causal\n                                        ).transpose(1, 2)\n                                    difference = actual.float() - cudnn.float()\n                                    relative_error = (\n                                        difference.square().mean().sqrt() / reference\n                                    )\n                                    self.assertLess(float(relative_error), 0.0001)\n\n    def test_sdpa_mask_and_dimension_fallback(self):\n        from transformers.integrations.sdpa_attention import sdpa_attention_forward\n\n        module = SimpleNamespace(num_key_value_groups=1, is_causal=True)\n        for tokens, dim, masked in [(4, 16, True), (5, 16, False), (4, 32, False)]:\n            q, k, v = [\n                torch.randn(3, 7, tokens, dim, device="cuda", dtype=torch.bfloat16)\n                for _ in range(3)\n            ]\n            mask = (\n                torch.ones(tokens, tokens, device="cuda", dtype=torch.bool).tril()\n                if masked\n                else None\n            )\n            actual = _mimo_local_attention_forward(\n                module, q, k, v, mask, is_causal=False\n            )[0]\n            expected = sdpa_attention_forward(module, q, k, v, mask, is_causal=False)[0]\n            torch.testing.assert_close(actual, expected, rtol=0, atol=0)\n\n    def test_transformers_local_model_and_mask_registration(self):\n        from transformers import Qwen2Config\n        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model\n\n        config = Qwen2Config(\n            vocab_size=32,\n            hidden_size=112,\n            intermediate_size=256,\n            num_hidden_layers=2,\n            num_attention_heads=7,\n            num_key_value_heads=7,\n            head_dim=16,\n        )\n        config._attn_implementation = "sdpa"\n        control = Qwen2Model(config).cuda().bfloat16().eval()\n        candidate_config = copy.deepcopy(config)\n        candidate_config._attn_implementation = _register_mimo_local_attention()\n        candidate = Qwen2Model(candidate_config).cuda().bfloat16().eval()\n        candidate.load_state_dict(control.state_dict())\n        x = torch.randn(3, 4, 112, device="cuda", dtype=torch.bfloat16)\n        for causal in [False, True]:\n            with (\n                torch.no_grad(),\n                patch(\n                    "sglang.kernels.ops.attention.mimo_local_attention.mimo_local_attention",\n                    wraps=mimo_local_attention,\n                ) as kernel,\n            ):\n                actual = candidate(inputs_embeds=x, is_causal=causal).last_hidden_state\n                if torch.cuda.get_device_capability()[0] == 9:\n                    self.assertEqual(kernel.call_count, 2)\n                expected = control(inputs_embeds=x, is_causal=causal).last_hidden_state\n                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'

old_namespace = {'torch': torch}
exec(OLD_FORWARD, old_namespace)
old_forward = old_namespace['_mimo_local_attention_forward']
old_name = 'sglang_mimo_before_cleanup_validation'
AttentionInterface.register(old_name, old_forward)
AttentionMaskInterface.register(old_name, ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])

# Run the original numerical/strided/CUDA Graph coverage unchanged, outside the PR.
historical_namespace = {'__name__': 'archived_mimo_validation'}
exec(ARCHIVED_TEST, historical_namespace)
suite = unittest.TestSuite([historical_namespace['TestMiMoLocalAttention']('test_shapes_strides_causality_and_graph')])
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()

model_cases = 0
fast_calls = 0
fallback_cases = 0
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for dim, tokens in ((16, 1), (16, 4), (16, 5), (32, 4)):
        cfg = Qwen2Config(vocab_size=32, hidden_size=7*dim, intermediate_size=256, num_hidden_layers=2, num_attention_heads=7, num_key_value_heads=7, head_dim=dim)
        cfg._attn_implementation = old_name
        control = Qwen2Model(cfg).cuda().to(dtype).eval()
        candidate_cfg = copy.deepcopy(cfg)
        candidate_cfg._attn_implementation = _register_mimo_local_attention(candidate_cfg, tokens)
        candidate = Qwen2Model(candidate_cfg).cuda().to(dtype).eval()
        candidate.load_state_dict(control.state_dict())
        x = torch.randn(3, tokens, 7*dim, device='cuda', dtype=dtype)
        for causal in (False, True):
            with torch.no_grad(), patch('sglang.kernels.ops.attention.mimo_local_attention.mimo_local_attention', wraps=mimo_local_attention) as kernel:
                actual = candidate(inputs_embeds=x, is_causal=causal).last_hidden_state
                count = kernel.call_count
                expect_fast = dim == 16 and tokens <= 4 and dtype != torch.float32
                assert count == (2 if expect_fast else 0), (dtype, dim, tokens, causal, count)
                expected = control(inputs_embeds=x, is_causal=causal).last_hidden_state
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            model_cases += 1
            fast_calls += count
            fallback_cases += not expect_fast

# Dynamic options still route to the same SDPA function and preserve kwargs.
module = SimpleNamespace(is_causal=True, num_key_value_groups=1)
q, k, v = [torch.randn(3,7,4,16,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
options = [dict(attention_mask=torch.ones(4,4,device='cuda',dtype=torch.bool).tril()), dict(attention_mask=None,dropout=0.1), dict(attention_mask=None,position_bias=torch.zeros(4,4,device='cuda')), dict(attention_mask=None,output_attentions=True)]
from transformers.integrations.sdpa_attention import sdpa_attention_forward
for opts in options:
    with patch('transformers.integrations.sdpa_attention.sdpa_attention_forward', wraps=sdpa_attention_forward) as fallback:
        torch.manual_seed(41)
        actual = _mimo_local_attention_forward(module,q,k,v,**opts)[0]
        assert fallback.call_count == 1
    torch.manual_seed(41)
    expected = old_forward(module,q,k,v,**opts)[0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

cfg = Qwen2Config(head_dim=16)
with patch('sglang.srt.models.mimo_audio.is_sm90_supported',return_value=False):
    assert _register_mimo_local_attention(cfg,4) == 'sdpa'
print('MIMO_CLEANUP_VALIDATION ' + json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'model_cases_bitwise_equal':model_cases,'actual_fast_kernel_calls':fast_calls,'model_fallback_cases':fallback_cases,'dynamic_sdpa_options':len(options),'archived_kernel_graph_test':'passed','hardware_selection':'non-Hopper branch emulated; actual execution on Hopper','kernel_source':'unchanged from 955d63e6ae1bce4e5f34c7561ad5a81a0c50b82e'}),flush=True)
