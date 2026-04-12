import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity import (
    build_sampling_kwargs,
    build_server_kwargs,
)


class TestCompareDiffusionTrajectorySimilarity(unittest.TestCase):
    def test_build_sampling_kwargs_includes_image_path(self):
        args = SimpleNamespace(
            prompt="A cat walking",
            width=832,
            height=480,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
            return_trajectory_decoded=False,
            num_frames=17,
            fps=8,
            image_path="/tmp/cat.png",
            guidance_scale_2=None,
        )

        sampling_kwargs = build_sampling_kwargs(args)

        self.assertEqual(sampling_kwargs["image_path"], "/tmp/cat.png")
        self.assertEqual(sampling_kwargs["num_frames"], 17)
        self.assertEqual(sampling_kwargs["fps"], 8)
        self.assertEqual(sampling_kwargs["prompt"], "A cat walking")

    def test_build_server_kwargs_includes_image_encoder_cpu_offload(self):
        args = SimpleNamespace(
            model_path="/tmp/model",
            model_id=None,
            backend="sglang",
            num_gpus=4,
            dit_cpu_offload=False,
            dit_layerwise_offload=True,
            text_encoder_cpu_offload=True,
            image_encoder_cpu_offload=True,
            vae_cpu_offload=True,
            pin_cpu_memory=False,
            enable_cfg_parallel=False,
            ulysses_degree=2,
            ring_degree=2,
            sp_degree=None,
            reference_component_path=[],
            reference_transformer_path=None,
        )

        server_kwargs = build_server_kwargs(args, variant="reference")

        self.assertTrue(server_kwargs["image_encoder_cpu_offload"])
        self.assertTrue(server_kwargs["dit_layerwise_offload"])
        self.assertEqual(server_kwargs["ring_degree"], 2)


if __name__ == "__main__":
    unittest.main()
