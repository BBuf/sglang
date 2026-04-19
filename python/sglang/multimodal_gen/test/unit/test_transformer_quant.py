"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)

from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageArchConfig,
    QwenImageDitConfig,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from sglang.multimodal_gen.tools.build_modelopt_fp8_transformer import (
    build_modelopt_fp8_transformer,
    get_default_keep_bf16_patterns,
    should_keep_bf16,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _updated_quant_config,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


class TestTransformerQuantHelpers(unittest.TestCase):
    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            tp_size=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_safetensors_to_load_prefers_mixed_export(
        self, _mock_download
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        _mock_download,
        mock_metadata,
        _mock_quant_metadata,
        mock_maybe_download,
        _mock_nvfp4,
    ):
        mock_maybe_download.side_effect = AssertionError(
            "local safetensors path should not trigger maybe_download_model"
        )
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            nunchaku_config = NunchakuConfig(transformer_weights_path=f.name)
            server_args = self._make_server_args(
                transformer_weights_path=nunchaku_config.transformer_weights_path,
                nunchaku_config=nunchaku_config,
            )

            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[nunchaku_config.transformer_weights_path],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)
        mock_maybe_download.assert_not_called()

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_prepare_nvfp4_weight_bytes_swaps_nibbles(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True)

        self.assertEqual(prepared.tolist(), [[0xBA, 0x01]])

    def test_prepare_nvfp4_weight_bytes_can_skip_nibble_swap(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=False)

        self.assertEqual(prepared.tolist(), [[0xAB, 0x10]])

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_flat_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "swap_weight_nibbles": False,
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_nested_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "swap_weight_nibbles": False,
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_builder_adds_diffusers_quant_type_for_nvfp4(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=["single_transformer_blocks.*.proj_mlp*"],
            swap_weight_nibbles=False,
        )

        self.assertEqual(updated["quantization_config"]["quant_type"], "NVFP4")
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["single_transformer_blocks.*.proj_mlp*"],
        )

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_flux_single_transformer_block_modelopt_excludes_use_full_prefix(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "single_transformer_blocks.*.proj_mlp*",
                "single_transformer_blocks.*.proj_out*",
                "single_transformer_blocks.*.attn.to_q",
            ],
        )

        block = FluxSingleTransformerBlock(
            dim=64,
            num_attention_heads=4,
            attention_head_dim=16,
            mlp_ratio=2.0,
            quant_config=quant_config,
            prefix="single_transformer_blocks.0",
        )

        self.assertEqual(block.proj_mlp.prefix, "single_transformer_blocks.0.proj_mlp")
        self.assertEqual(block.proj_out.prefix, "single_transformer_blocks.0.proj_out")
        self.assertEqual(
            block.attn.to_q.prefix, "single_transformer_blocks.0.attn.to_q"
        )
        self.assertIsInstance(block.proj_mlp.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_qwen_image_block_modelopt_fp8_uses_full_prefixes(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=[
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.add_q_proj",
                "transformer_blocks.0.img_mlp.net.0.proj",
                "transformer_blocks.0.img_mlp.net.2",
                "transformer_blocks.0.img_mod",
            ],
        )

        block = QwenImageTransformerBlock(
            dim=64,
            num_attention_heads=4,
            attention_head_dim=16,
            quant_config=quant_config,
            prefix="transformer_blocks.0",
        )

        self.assertEqual(block.attn.to_q.prefix, "transformer_blocks.0.attn.to_q")
        self.assertEqual(
            block.attn.add_q_proj.prefix, "transformer_blocks.0.attn.add_q_proj"
        )
        self.assertEqual(
            block.img_mlp.net[0].proj.prefix,
            "transformer_blocks.0.img_mlp.net.0.proj",
        )
        self.assertEqual(
            block.img_mlp.net[2].prefix, "transformer_blocks.0.img_mlp.net.2"
        )
        self.assertEqual(block.img_mod[1].prefix, "transformer_blocks.0.img_mod")
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(
            block.attn.add_q_proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(
            block.img_mlp.net[0].proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(
            block.img_mlp.net[2].quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(block.img_mod[1].quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_k.quant_method, ModelOptFp8LinearMethod)

    @patch(
        "sglang.multimodal_gen.runtime.models.dits.qwen_image.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_qwen_image_top_level_modelopt_fp8_uses_full_prefixes(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
        _mock_device,
    ):
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=["img_in", "txt_in", "proj_out"],
        )
        arch_config = QwenImageArchConfig(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(2, 2, 4),
        )

        model = QwenImageTransformer2DModel(
            config=QwenImageDitConfig(arch_config=arch_config),
            hf_config={},
            quant_config=quant_config,
        )

        self.assertEqual(model.img_in.prefix, "img_in")
        self.assertEqual(model.txt_in.prefix, "txt_in")
        self.assertEqual(model.proj_out.prefix, "proj_out")
        self.assertIsInstance(model.img_in.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(model.txt_in.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(model.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(
            model.transformer_blocks[0].attn.to_q.quant_method,
            ModelOptFp8LinearMethod,
        )

    def test_qwen_image_default_fp8_fallbacks_cover_shared_t2i_and_edit_layers(self):
        patterns = get_default_keep_bf16_patterns(
            model_type="qwen-image", class_name=None
        )
        auto_patterns = get_default_keep_bf16_patterns(
            model_type="auto", class_name="QwenImageTransformer2DModel"
        )

        self.assertEqual(patterns, auto_patterns)
        self.assertTrue(should_keep_bf16("img_in.weight", patterns))
        self.assertTrue(should_keep_bf16("txt_in.weight", patterns))
        self.assertTrue(should_keep_bf16("proj_out.weight", patterns))
        self.assertTrue(
            should_keep_bf16(
                "time_text_embed.timestep_embedder.linear_1.weight", patterns
            )
        )
        self.assertTrue(
            should_keep_bf16("transformer_blocks.0.img_mod.1.weight", patterns)
        )
        self.assertTrue(
            should_keep_bf16("transformer_blocks.0.txt_mod.1.weight", patterns)
        )
        self.assertTrue(
            should_keep_bf16("transformer_blocks.0.img_mlp.net.2.weight", patterns)
        )
        self.assertFalse(
            should_keep_bf16("transformer_blocks.0.attn.to_q.weight", patterns)
        )
        self.assertFalse(
            should_keep_bf16("transformer_blocks.0.img_mlp.net.0.proj.weight", patterns)
        )
        self.assertFalse(
            should_keep_bf16("transformer_blocks.0.txt_mlp.net.2.weight", patterns)
        )

    def test_modelopt_fp8_builder_writes_bf16_fallback_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = f"{tmpdir}/source"
            base_dir = f"{tmpdir}/base"
            output_dir = f"{tmpdir}/output"
            ckpt_path = f"{tmpdir}/transformer.pt"
            os.makedirs(source_dir)
            os.makedirs(base_dir)

            config = {
                "_class_name": "QwenImageTransformer2DModel",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": [],
                },
            }
            for model_dir in (source_dir, base_dir):
                with open(f"{model_dir}/config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f)

            weight_name = "transformer_blocks.0.attn.to_q.weight"
            base_weight = torch.full((2, 2), 0.125, dtype=torch.bfloat16)
            source_weight = torch.full((2, 2), 8.0, dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            save_file({weight_name: source_weight}, f"{source_dir}/model.safetensors")
            save_file({weight_name: base_weight}, f"{base_dir}/model.safetensors")
            torch.save(
                {
                    "model_state_dict": {
                        "transformer_blocks.0.attn.to_q.weight_quantizer._amax": torch.tensor(
                            448.0
                        ),
                        "transformer_blocks.0.attn.to_q.input_quantizer._amax": torch.tensor(
                            448.0
                        ),
                    }
                },
                ckpt_path,
            )

            stats = build_modelopt_fp8_transformer(
                modelopt_hf_dir=source_dir,
                modelopt_backbone_ckpt=ckpt_path,
                base_transformer_dir=base_dir,
                output_dir=output_dir,
                model_type="none",
                keep_bf16_patterns=[r"^transformer_blocks\.0\.attn\.to_q$"],
            )

            self.assertEqual(stats["bf16_fallback_weights"], 1)
            self.assertEqual(stats["quantized_weights"], 0)
            with safe_open(
                f"{output_dir}/model.safetensors", framework="pt", device="cpu"
            ) as f:
                keys = set(f.keys())
                output_weight = f.get_tensor(weight_name)

            self.assertEqual(output_weight.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(output_weight, base_weight))
            self.assertNotIn("transformer_blocks.0.attn.to_q.weight_scale", keys)
            self.assertNotIn("transformer_blocks.0.attn.to_q.input_scale", keys)


if __name__ == "__main__":
    unittest.main()
