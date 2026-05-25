import unittest

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    _filter_invalid_block_quant_configs,
)


class TestMoeTritonConfig(unittest.TestCase):
    def test_filter_invalid_block_quant_configs(self):
        configs = {
            1: {"BLOCK_SIZE_K": 32},
            2: {"BLOCK_SIZE_K": 64},
            3: {"BLOCK_SIZE_K": 96},
            4: {"BLOCK_SIZE_K": 128},
            5: {"BLOCK_SIZE_K": 0},
            8: {"BLOCK_SIZE_K": 256},
        }

        self.assertEqual(
            _filter_invalid_block_quant_configs(configs, 128, "dummy.json"),
            {
                1: {"BLOCK_SIZE_K": 32},
                2: {"BLOCK_SIZE_K": 64},
                4: {"BLOCK_SIZE_K": 128},
            },
        )

    def test_filter_invalid_block_quant_configs_returns_none_if_empty(self):
        self.assertIsNone(
            _filter_invalid_block_quant_configs(
                {4: {"BLOCK_SIZE_K": 256}}, 128, "dummy.json"
            )
        )

    def test_filter_invalid_block_quant_configs_ignores_unblocked_quant(self):
        configs = {4: {"BLOCK_SIZE_K": 256}}
        self.assertIs(
            _filter_invalid_block_quant_configs(configs, 0, "dummy.json"), configs
        )


if __name__ == "__main__":
    unittest.main()
