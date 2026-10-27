"""Compare native CPU preprocessing/scheduler inputs without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLProcessor

p = argparse.ArgumentParser()
p.add_argument('arm', choices=['sgl', 'omni'])
p.add_argument('--contract', type=Path, default=Path('/task/inputs/joy-workloads.json'))
p.add_argument('--label', default='')
a = p.parse_args()
contract = json.loads(a.contract.read_text())
model = Path(contract['model_path'])
image = Image.open(Path('/task')/contract['reference_image']['path']).convert('RGB')
processor = Qwen3VLProcessor.from_pretrained(model/'processor', local_files_only=True)
if a.arm == 'sgl':
    from sglang.multimodal_gen.configs.pipeline_configs.joy_image import JoyImageEditPipelineConfig
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    config = JoyImageEditPipelineConfig()
    resized, size = config.preprocess_condition_image(image, contract['width'], contract['height'], None)
    def format_text(prompt):
        batch = SimpleNamespace(prompt=prompt, negative_prompt='', condition_image=image)
        return config.prepare_image_processor_kwargs(batch)['text']
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model/'scheduler', local_files_only=True)
    sigmas = config.prepare_sigmas(None, contract['num_inference_steps'])
    if sigmas is None:
        scheduler.set_timesteps(contract['num_inference_steps'], device='cpu')
    else:
        scheduler.set_timesteps(sigmas=sigmas, device='cpu')
    noise_shape = (1, 16, 1, contract['height']//8, contract['width']//8)
    vae_mode = config.vae_config.encode_sample_mode()
else:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JOY_PROMPT_TEMPLATE, _format_qwen_multimodal_prompt, _find_best_bucket, _resize_center_crop)
    h, w = _find_best_bucket(contract['height'], contract['width'])
    resized = _resize_center_crop(image, h, w)
    size = (w, h)
    def format_text(prompt):
        return [JOY_PROMPT_TEMPLATE.format(_format_qwen_multimodal_prompt(prompt))]
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model/'scheduler', local_files_only=True)
    scheduler.set_timesteps(contract['num_inference_steps'], device='cpu')
    noise_shape = (1, 1, 16, 1, h//8, w//8)
    vae_mode = 'sample; _encode_vae_image does not forward request generator'

def signature(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), sha256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())

rows=[]
for prompt in contract['prompts'] + [{'id':'negative','prompt':''}]:
    text = format_text(prompt['prompt'])
    inputs = processor(text=text, images=[resized], padding=True, return_tensors='pt')
    rows.append(dict(id=prompt['id'], text=text, tensors={k:signature(v) for k,v in inputs.items() if torch.is_tensor(v)}))
noise = torch.randn(noise_shape, generator=torch.Generator('cpu').manual_seed(contract['seed']), dtype=torch.bfloat16)
assert size == (contract['width'], contract['height']), 'Requested size is not a native aligned bucket'
result = dict(arm=a.arm, cpu_preflight_only=True, contract_sha256=hashlib.sha256(a.contract.read_bytes()).hexdigest(), reference_sha256=hashlib.sha256((Path('/task')/contract['reference_image']['path']).read_bytes()).hexdigest(), resized_size=size,
              resized_pixel_sha256=hashlib.sha256(np.asarray(resized).tobytes()).hexdigest(),
              prompts=rows, initial_noise=signature(noise.flatten()), timesteps=signature(scheduler.timesteps),
              sigmas=signature(scheduler.sigmas), sigma_values=scheduler.sigmas.tolist(), vae_sampling_semantics=vae_mode)
label = a.label+'-' if a.label else ''
path=Path(f'/task/evidence/preflight/joy-{label}{a.arm}-inputs.json')
path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'arm':a.arm,'artifact':str(path),'noise':result['initial_noise'],'timesteps':result['timesteps']}))
