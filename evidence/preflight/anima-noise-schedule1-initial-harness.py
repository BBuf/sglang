"""Compare native CPU random draw and scheduler functions before inference."""
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path('/task')
out = root / 'outputs/diagnostics/anima-noise-schedule1'
out.mkdir(parents=True, exist_ok=True)
arm = sys.argv[1]
if arm == 'compare':
    a, b = [torch.load(out / (name + '.pt'), weights_only=True) for name in ['sgl', 'omni']]
    comparisons = {key: dict(exact=torch.equal(a[key], b[key]), shape=list(a[key].shape),
        dtype=str(a[key].dtype), max_abs=float((a[key] - b[key]).abs().max())) for key in a}
    result = dict(scope='Native CPU noise and scheduler functions, not actual full-model runtime inputs',
        comparisons=comparisons, all_exact=all(x['exact'] for x in comparisons.values()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = root / 'evidence/preflight/anima-noise-schedule1.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result)); assert result['all_exact']; sys.exit()

assert arm in ['sgl', 'omni'] and not (out / (arm + '.pt')).exists()
config_path = root / 'cache/common/Anima-Base-v1.0-Diffusers/scheduler/scheduler_config.json'
config = json.loads(config_path.read_text())
generator = torch.Generator(device='cpu').manual_seed(42)
if arm == 'sgl':
    from sglang.multimodal_gen.configs.pipeline_configs.anima import AnimaPipelineConfig
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import randn_tensor
    pipeline = AnimaPipelineConfig()
    batch = SimpleNamespace(height=1024, width=1024)
    shape = pipeline.prepare_latent_shape(batch, 1, 1)
    latents = randn_tensor(shape, generator=generator, device=torch.device('cpu'), dtype=pipeline.get_latent_dtype(torch.bfloat16))
    latents = pipeline.maybe_pack_latents(latents, 1, batch)
    scheduler = FlowMatchEulerDiscreteScheduler(**{k: v for k, v in config.items() if not k.startswith('_')})
    scheduler.set_timesteps(50, device='cpu', sigmas=pipeline.prepare_sigmas(None, 50))
    functions = [AnimaPipelineConfig, FlowMatchEulerDiscreteScheduler, randn_tensor]
else:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    pipeline.scheduler = scheduler
    latents = pipeline.prepare_latents(1, 16, 1024, 1024, torch.float32, torch.device('cpu'), generator, None)
    pipeline.prepare_timesteps(50, device=torch.device('cpu'))
    functions = [AnimaPipeline, FlowMatchEulerDiscreteScheduler]
assert latents.shape == (1, 16, 1, 128, 128) and latents.dtype == torch.float32
values = dict(latents=latents, timesteps=scheduler.timesteps, sigmas=scheduler.sigmas)
torch.save(values, out / (arm + '.pt'))
sources = {Path(inspect.getfile(f)) for f in functions}
meta = dict(arm=arm, torch=str(torch.__version__), effective_scheduler_config=dict(scheduler.config),
    tensor_hashes={k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in values.items()},
    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out / (arm + '.json')).write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta))
