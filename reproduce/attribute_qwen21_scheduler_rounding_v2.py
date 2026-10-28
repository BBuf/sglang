"""CPU diagnostic: attribute native schedule differences without changing serving code."""
import hashlib
import json
import math
import numpy as np
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
for label in ['native_sgl', 'diagnostic_math_exp_only', 'diagnostic_math_exp_and_numpy_stretch']:
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model / 'scheduler', local_files_only=True)
    settings = scheduler.config
    mu = calculate_linear_shift(4096, base_seq_len=settings.get('base_image_seq_len', 256),
        max_seq_len=settings.get('max_image_seq_len', 4096), base_shift=settings.get('base_shift', .5),
        max_shift=settings.get('max_shift', 1.15))
    if label != 'native_sgl':
        scheduler._time_shift_exponential = lambda mu, sigma, t: math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    shift_dtype = str(scheduler.time_shift(mu, 1.0, np.array([1.0, .5, .02], dtype=np.float32)).dtype)
    if label == 'diagnostic_math_exp_and_numpy_stretch':
        scheduler.stretch_shift_to_terminal = numpy_stretch
    scheduler.set_timesteps(sigmas=config.prepare_sigmas(None, 50), device='cpu', mu=mu)
    outputs[label] = dict(exponential_shift_numpy_dtype=shift_dtype, sigmas_sha256=signature(scheduler.sigmas),
                          timesteps_sha256=signature(scheduler.timesteps), mu=mu,
                          shift_terminal=settings.shift_terminal)
assert outputs['native_sgl']['sigmas_sha256'] == native['sgl']['sigmas']['sha256']
assert outputs['native_sgl']['timesteps_sha256'] == native['sgl']['timesteps']['sha256']
matches = {label: {key: outputs[label][key + '_sha256'] == native['omni'][key]['sha256'] for key in ['sigmas', 'timesteps']} for label in outputs if label != 'native_sgl'}
result = dict(cpu_only=True, serving_code_changed=False, outputs=outputs,
              controlled_variants_reproduce_native_omni_hashes=matches,
              interpretation='The original stretch-only probe did not change the result. This probe changes only the CPU exponential-shift scalar math.exp vs np.exp, with a separate stretch control. See exact native-hash matches and intermediate dtype evidence. No serving code or request/steps modified; full-model RGB/alpha quality remains required.')
target = folder / 'qwen21-scheduler-rounding-attribution-v2.json'
assert not target.exists()
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
assert all(matches['diagnostic_math_exp_only'].values()), 'Rounding cause not fully isolated; inspect other native differences'
