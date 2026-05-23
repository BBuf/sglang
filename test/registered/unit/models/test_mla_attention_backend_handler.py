import unittest

import torch

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _handle_attention_backend,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeForwardMode:
    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return False

    def is_draft_extend(self):
        return False

    def is_extend_without_speculative(self):
        return True


class _FakeAttn:
    flashinfer_mla_disable_ragged = True
    chunked_prefix_cache_threshold = 0
    disable_chunked_prefix_cache = False


class _FakeForwardBatch:
    def __init__(self, seq_lens_cpu, use_tensor=False, max_chunk_capacity=4096):
        self.forward_mode = _FakeForwardMode()
        self.seq_lens_cpu = (
            torch.tensor(seq_lens_cpu, dtype=torch.int32)
            if use_tensor
            else seq_lens_cpu
        )
        self.extend_prefix_lens_cpu = [0] * len(seq_lens_cpu)
        self.max_chunk_capacity = max_chunk_capacity

    def get_max_chunk_capacity(self):
        return self.max_chunk_capacity


class _FakeServerArgs:
    flashinfer_mla_disable_ragged = True
    prefill_attention_backend = "flashinfer"
    speculative_attention_mode = "prefill"


class _FakeModelRunner:
    kv_cache_dtype = torch.bfloat16
    server_args = _FakeServerArgs()


class TestMlaAttentionBackendHandler(unittest.TestCase):
    def test_flashinfer_short_uncached_prefill_uses_paged_mla(self):
        forward_batch = _FakeForwardBatch([1000])

        method = _handle_attention_backend(_FakeAttn(), forward_batch, "flashinfer")

        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_flashinfer_long_uncached_prefill_uses_mha(self):
        forward_batch = _FakeForwardBatch(
            [8000], use_tensor=True, max_chunk_capacity=20000
        )

        method = _handle_attention_backend(_FakeAttn(), forward_batch, "flashinfer")

        self.assertEqual(method, AttnForwardMethod.MHA_CHUNKED_KV)

    def test_hybrid_routes_long_uncached_prefill_to_fallback_backend(self):
        prefill_backend = object()
        decode_backend = object()
        long_prefill_backend = object()
        hybrid_backend = HybridAttnBackend(
            _FakeModelRunner(),
            prefill_backend=prefill_backend,
            decode_backend=decode_backend,
            long_prefill_backend=long_prefill_backend,
        )

        short_batch = _FakeForwardBatch([1000])
        long_batch = _FakeForwardBatch([8000], use_tensor=True)

        self.assertIs(
            hybrid_backend._select_backend(short_batch.forward_mode, short_batch),
            prefill_backend,
        )
        self.assertIs(
            hybrid_backend._select_backend(long_batch.forward_mode, long_batch),
            long_prefill_backend,
        )


if __name__ == "__main__":
    unittest.main()
