"""CPU diagnostic: attribute native schedule differences without changing serving code."""
import hashlib
import json
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import calculate_linear_shift

root = Path('/task')
folder = root / 'evidence/preflight'
model = root / 'cache/common/Qwen-Image-2.1'
native = {a: json.loads((folder / f'qwen21-base-t2i-{a}-inputs.json').read_text()) for a in ['sgl', 'omni']}


def signature(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def numpy_stretch(tensor):
    # The actual installed Omni scheduler applies this formula to a NumPy array;
    # the native SGL scheduler passes a float32 Torch tensor to its same formula.
    values = tensor.numpy()
    one_minus_z = 1 - values
    scale_factor = one_minus_z[-1] / (1 - scheduler.config.shift_terminal)
    return torch.from_numpy(1 - one_minus_z / scale_factor)


config = QwenImage21PipelineConfig()
outputs = {}
for label in ['native_sgl', 'diagnostic_numpy_stretch_only']:
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model / 'scheduler', local_files_only=True)
    settings = scheduler.config
    mu = calculate_linear_shift(4096, base_seq_len=settings.get('base_image_seq_len', 256),
        max_seq_len=settings.get('max_image_seq_len', 4096), base_shift=settings.get('base_shift', .5),
        max_shift=settings.get('max_shift', 1.15))
    if label != 'native_sgl':
        scheduler.stretch_shift_to_terminal = numpy_stretch
    scheduler.set_timesteps(sigmas=config.prepare_sigmas(None, 50), device='cpu', mu=mu)
    outputs[label] = dict(sigmas_sha256=signature(scheduler.sigmas),
                          timesteps_sha256=signature(scheduler.timesteps), mu=mu,
                          shift_terminal=settings.shift_terminal)
assert outputs['native_sgl']['sigmas_sha256'] == native['sgl']['sigmas']['sha256']
assert outputs['native_sgl']['timesteps_sha256'] == native['sgl']['timesteps']['sha256']
matches = {key: outputs['diagnostic_numpy_stretch_only'][key + '_sha256'] == native['omni'][key]['sha256']
           for key in ['sigmas', 'timesteps']}
result = dict(cpu_only=True, serving_code_changed=False, outputs=outputs,
              changing_only_cpu_stretch_arithmetic_reproduces_native_omni_hashes=matches,
              interpretation='Same fifty-step grid and BF16 workload. SGL stretches the schedule with Torch float32, Omni with NumPy float32; native sigma differences are at most 5.96e-8 (two float32 ULPs). Keep both native implementations and disclose this difference; full-model RGB/alpha quality is still required before performance comparison.')
target = folder / 'qwen21-scheduler-rounding-attribution.json'
assert not target.exists()
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
assert all(matches.values()), 'Rounding cause not fully isolated; inspect other native differences'
