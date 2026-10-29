"""Resolve pinned native pipeline/config classes without loading any weights."""
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import LatentPreparationStage

ROOT = Path("/task")
META = ROOT / "evidence/downloads/sana-video-pinned-metadata"
rows = []
for variant, height, width in [("480p", 480, 832), ("720p", 704, 1280)]:
    folder = META / variant
    model_id = "Efficient-Large-Model/SANA-Video_2B_" + variant + "_diffusers"
    info = get_model_info(str(folder), backend="sglang", model_id=model_id)
    pipeline = info.pipeline_config_cls()
    vae = json.loads((folder / "vae/config.json").read_text())
    transformer = json.loads((folder / "transformer/config.json").read_text())
    pipeline.vae_config.update_model_arch(vae)
    pipeline.vae_config.post_init()
    pipeline.dit_config.update_model_arch(transformer)
    batch = SimpleNamespace(height=height, width=width, num_frames=81)
    frames = LatentPreparationStage.adjust_video_length(None, batch, SimpleNamespace(pipeline_config=pipeline))
    actual = list(pipeline.prepare_latent_shape(batch, 1, frames))
    temporal, spatial = (4, 8) if variant == "480p" else (8, 32)
    expected = [1, transformer["in_channels"], (81 - 1) // temporal + 1, height // spatial, width // spatial]
    rows.append(dict(variant=variant, pipeline_class=info.pipeline_cls.__name__, pipeline_config_class=type(pipeline).__name__, vae_config_class=type(pipeline.vae_config).__name__, checkpoint_vae_class=vae["_class_name"], effective_compression=[pipeline.vae_config.arch_config.temporal_compression_ratio, pipeline.vae_config.arch_config.spatial_compression_ratio], checkpoint_compression=[temporal, spatial], native_latent_shape=actual, checkpoint_latent_shape=expected, shape_matches=actual == expected))
result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), scope="Native frozen registry/config and actual latent-shape methods; no weights, no GPU, no full inference. Config-loader path update_model_arch/post_init reproduced with original metadata.", rows=rows, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = ROOT / "evidence/preflight/sana-video-native-config1.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
