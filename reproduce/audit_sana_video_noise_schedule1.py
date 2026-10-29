"""Native CPU schedule/noise audit before checkpoint downloads or GPU inference."""
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path("/task")
META = ROOT / "evidence/downloads/sana-video-pinned-metadata"
OUT = ROOT / "outputs/diagnostics/sana-video-noise-schedule1"
OUT.mkdir(parents=True, exist_ok=True)
arm = sys.argv[1]
if arm == "compare":
    a, b = [torch.load(OUT / (name + ".pt"), weights_only=True) for name in ["sgl", "omni"]]
    rows = {key: dict(exact=torch.equal(a[key], b[key]), shape=list(a[key].shape), dtype=str(a[key].dtype), max_abs=float((a[key] - b[key]).abs().max())) for key in a}
    result = dict(scope="Pinned 480p checkpoint, 832x480,81frames,50steps,CPU seed1101. Native CPU noise/schedule only, not full-model runtime equivalence.", comparisons=rows, all_exact=all(row["exact"] for row in rows.values()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest = ROOT / "evidence/preflight/sana-video-noise-schedule1.json"
    assert not dest.exists()
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert result["all_exact"]
    sys.exit()

assert arm in ["sgl", "omni"] and not (OUT / (arm + ".pt")).exists()
config = json.loads((META / "480p/scheduler/scheduler_config.json").read_text())
generator = torch.Generator(device="cpu").manual_seed(1101)
if arm == "sgl":
    from sglang.multimodal_gen.configs.pipeline_configs.sana_video import SanaVideoPipelineConfig
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_dpm_solver_multistep import DPMSolverMultistepScheduler
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import LatentPreparationStage, randn_tensor
    pipeline = SanaVideoPipelineConfig()
    pipeline.vae_config.update_model_arch(json.loads((META / "480p/vae/config.json").read_text()))
    pipeline.vae_config.post_init()
    pipeline.dit_config.update_model_arch(json.loads((META / "480p/transformer/config.json").read_text()))
    batch = SimpleNamespace(height=480, width=832, num_frames=81)
    args = SimpleNamespace(pipeline_config=pipeline)
    latent_frames = LatentPreparationStage.adjust_video_length(None, batch, args)
    shape = pipeline.prepare_latent_shape(batch, 1, latent_frames)
    latents = randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=pipeline.get_latent_dtype(torch.bfloat16))
    scheduler = DPMSolverMultistepScheduler(**{key: value for key, value in config.items() if not key.startswith("_")})
    scheduler.set_shift(pipeline.flow_shift)
    scheduler.set_timesteps(50, device="cpu", sigmas=pipeline.prepare_sigmas(None, 50))
    functions = [SanaVideoPipelineConfig, DPMSolverMultistepScheduler, LatentPreparationStage, randn_tensor]
else:
    from diffusers.schedulers import DPMSolverMultistepScheduler
    from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import SanaVideoPipeline, retrieve_timesteps
    pipeline = SanaVideoPipeline.__new__(SanaVideoPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    latents = pipeline.prepare_latents(1, 16, 480, 832, 81, torch.float32, torch.device("cpu"), generator, None)
    scheduler = DPMSolverMultistepScheduler.from_config(config)
    retrieve_timesteps(scheduler, 50, torch.device("cpu"), None, None)
    functions = [SanaVideoPipeline, DPMSolverMultistepScheduler, retrieve_timesteps]
assert latents.shape == (1, 16, 21, 60, 104) and latents.dtype == torch.float32
values = dict(latents=latents, timesteps=scheduler.timesteps, sigmas=scheduler.sigmas)
torch.save(values, OUT / (arm + ".pt"))
sources = {Path(inspect.getfile(function)) for function in functions}
record = dict(arm=arm, torch=str(torch.__version__), scheduler_config=dict(scheduler.config), tensor_hashes={key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in values.items()}, source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT / (arm + ".json")).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
