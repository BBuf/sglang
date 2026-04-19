import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImageEditPlusPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage import (
    QwenImageEditPlusSamplingParams,
    QwenImageSamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info


class TestQwenImageEditRegistry(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_base_edit_model_id_resolves_to_edit_pipeline(self):
        info = _get_config_info("/tmp/local-qwen-edit", model_id="Qwen-Image-Edit")

        self.assertIs(info.pipeline_config_cls, QwenImageEditPipelineConfig)
        self.assertIs(info.sampling_param_cls, QwenImageSamplingParams)

    def test_edit_plus_model_ids_resolve_to_plus_pipelines(self):
        info_2509 = _get_config_info(
            "/tmp/local-qwen-edit-2509", model_id="Qwen-Image-Edit-2509"
        )
        info_2511 = _get_config_info(
            "/tmp/local-qwen-edit-2511", model_id="Qwen-Image-Edit-2511"
        )

        self.assertIs(info_2509.pipeline_config_cls, QwenImageEditPlusPipelineConfig)
        self.assertIs(info_2509.sampling_param_cls, QwenImageEditPlusSamplingParams)
        self.assertIs(
            info_2511.pipeline_config_cls, QwenImageEditPlus_2511_PipelineConfig
        )
        self.assertIs(info_2511.sampling_param_cls, QwenImageEditPlusSamplingParams)


class TestQwenImageEditPromptImageFlow(unittest.TestCase):
    def test_base_edit_uses_single_image_prompt_template(self):
        config = QwenImageEditPipelineConfig()
        batch = SimpleNamespace(
            prompt="Make the cup matte white",
            negative_prompt="",
        )

        kwargs = config.prepare_image_processor_kwargs(batch)

        self.assertEqual(kwargs["padding"], True)
        self.assertEqual(len(kwargs["text"]), 1)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", kwargs["text"][0])
        self.assertIn("Make the cup matte white", kwargs["text"][0])

    def test_plus_single_prompt_can_condition_on_multiple_images(self):
        config = QwenImageEditPlusPipelineConfig()
        images = [object(), object()]
        batch = SimpleNamespace(
            prompt="Combine the two reference images",
            negative_prompt="",
            condition_image=images,
        )

        kwargs = config.prepare_image_processor_kwargs(batch)

        self.assertEqual(kwargs["per_prompt_images"], [images])
        self.assertIn("Picture 1:", kwargs["text"][0])
        self.assertIn("Picture 2:", kwargs["text"][0])
        self.assertIn("Combine the two reference images", kwargs["text"][0])

    def test_plus_multi_prompt_reuses_single_shared_image(self):
        config = QwenImageEditPlusPipelineConfig()
        shared_image = object()
        batch = SimpleNamespace(
            prompt=["Oil painting", "Watercolor"],
            negative_prompt="",
            condition_image=shared_image,
        )

        kwargs = config.prepare_image_processor_kwargs(batch)

        self.assertEqual(len(kwargs["text"]), 2)
        self.assertEqual(
            kwargs["per_prompt_images"],
            [[shared_image], [shared_image]],
        )

    def test_plus_multi_prompt_pairs_matching_image_list_one_to_one(self):
        config = QwenImageEditPlusPipelineConfig()
        image_a = object()
        image_b = object()
        batch = SimpleNamespace(
            prompt=["Edit image A", "Edit image B"],
            negative_prompt="",
            condition_image=[image_a, image_b],
        )

        kwargs = config.prepare_image_processor_kwargs(batch)

        self.assertEqual(kwargs["per_prompt_images"], [[image_a], [image_b]])
        self.assertIn("Edit image A", kwargs["text"][0])
        self.assertIn("Edit image B", kwargs["text"][1])

    def test_plus_rejects_ambiguous_prompt_image_cardinality(self):
        config = QwenImageEditPlusPipelineConfig()
        batch = SimpleNamespace(
            prompt=["Edit image A", "Edit image B"],
            negative_prompt="",
            condition_image=[object(), object(), object()],
        )

        with self.assertRaisesRegex(ValueError, "same number"):
            config.prepare_image_processor_kwargs(batch)


class TestQwenImageEditConditionFlow(unittest.TestCase):
    def test_base_edit_prepare_cond_kwargs_without_rotary(self):
        config = QwenImageEditPipelineConfig()
        config.vae_config.post_init()
        batch = SimpleNamespace(
            latents=torch.zeros(1, 4096, 64),
            height=512,
            width=512,
            original_condition_image_size=(512, 512),
        )
        prompt_embeds = [torch.zeros(1, 9, 4096)]

        kwargs = config._prepare_edit_cond_kwargs(
            batch=batch,
            prompt_embeds=prompt_embeds,
            rotary_emb=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(kwargs["txt_seq_lens"], [9])
        self.assertIsNone(kwargs["freqs_cis"])
        self.assertEqual(
            kwargs["img_shapes"],
            [[(1, 32, 32), (1, 64, 64)]],
        )

    def test_plus_preprocess_vae_image_tracks_per_image_vae_sizes(self):
        config = QwenImageEditPlusPipelineConfig()
        image_a = Image.new("RGB", (320, 160), color="white")
        image_b = Image.new("RGB", (160, 320), color="black")
        batch = SimpleNamespace(condition_image=[image_a, image_b])
        processor = MagicMock()
        processor.preprocess.side_effect = ["vae-a", "vae-b"]

        result = config.preprocess_vae_image(batch, processor)

        self.assertIs(result, batch)
        self.assertEqual(batch.vae_image, ["vae-a", "vae-b"])
        self.assertEqual(len(batch.vae_image_sizes), 2)
        for width, height in batch.vae_image_sizes:
            self.assertEqual(width % 16, 0)
            self.assertEqual(height % 16, 0)
            self.assertGreater(width * height, 0)
        processor.preprocess.assert_any_call(
            image_a, batch.vae_image_sizes[0][1], batch.vae_image_sizes[0][0]
        )
        processor.preprocess.assert_any_call(
            image_b, batch.vae_image_sizes[1][1], batch.vae_image_sizes[1][0]
        )

    def test_base_edit_postprocess_image_latent_repeats_shared_image_for_batch(self):
        config = QwenImageEditPipelineConfig()
        batch = SimpleNamespace(batch_size=2)
        latent_condition = torch.arange(
            1 * 16 * 1 * 16 * 16, dtype=torch.float32
        ).reshape(1, 16, 1, 16, 16)

        packed = config.postprocess_image_latent(latent_condition, batch)

        self.assertEqual(packed.shape, (2, 64, 64))
        self.assertTrue(torch.equal(packed[0], packed[1]))

    def test_base_edit_postprocess_image_latent_rejects_non_divisible_batch(self):
        config = QwenImageEditPipelineConfig()
        batch = SimpleNamespace(batch_size=3)
        latent_condition = torch.zeros(2, 16, 1, 16, 16)

        with self.assertRaisesRegex(ValueError, "Cannot duplicate"):
            config.postprocess_image_latent(latent_condition, batch)

    def test_base_edit_slices_condition_image_noise_tokens(self):
        config = QwenImageEditPipelineConfig()
        noise = torch.zeros(1, 10, 64)
        latents = torch.zeros(1, 4, 64)

        sliced = config.slice_noise_pred(noise, latents)

        self.assertEqual(sliced.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
