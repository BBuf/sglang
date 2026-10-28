"""CPU input audit only: real processor/scheduler/noise, no model or synthetic output."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Qwen3VLProcessor


def signature(tensor):
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(raw).hexdigest())


class ProcessorCaptured(Exception):
    pass


class CaptureProcessor:
    """Stop the native prompt method after executing its real processor on CPU."""
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.last = None

    def __call__(self, **kwargs):
        actual = self.processor(**kwargs)
        self.last = dict(text=kwargs['text'],
                         tensors={k:signature(v) for k,v in actual.items() if torch.is_tensor(v)})
        raise ProcessorCaptured


parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['sgl', 'omni'])
parser.add_argument('--contract', type=Path, required=True)
args = parser.parse_args()
contract = json.loads(args.contract.read_text())
assert contract['batch'] == contract['concurrency'] == 1
assert contract['width'] == contract['height'] == 1024, 'Other shapes require a new resize admission audit'
model = Path(contract['model_path'])
model_index = json.loads((model / 'model_index.json').read_text())
vae_config = json.loads((model / 'vae/config.json').read_text())
transformer_config = json.loads((model / 'transformer/config.json').read_text())
assert vae_config['in_channels'] == vae_config['out_channels'] == 4
assert vae_config['z_dim'] == transformer_config['in_channels'] == 64
if 'sigmas' in contract:
    assert model_index['sample_sigmas'] == contract['sigmas']
else:
    assert not model_index.get('sample_sigmas')
processor = Qwen3VLProcessor.from_pretrained(model / 'processor', local_files_only=True)
capture = CaptureProcessor(processor)
images = []
reference_sha = None
if contract['task'] == 'edit':
    reference = Path('/task') / contract['reference_image']['path']
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    assert reference_sha == contract['reference_image']['sha256']
    with Image.open(reference) as image:
        images = [image.convert('RGBA')]
generator = torch.Generator('cpu').manual_seed(contract['seed'])
steps = contract.get('num_inference_steps', len(contract.get('sigmas', [])))
sources = {}
vae_pixels = []
if args.arm == 'sgl':
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import calculate_linear_shift
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import QwenImage21EncodingStage
    config = QwenImage21PipelineConfig()
    config.sample_sigmas = model_index.get('sample_sigmas')
    # This resize/pixel chain is source-derived from EncodingStage.forward.
    # The native prompt processor method itself is executed below, stopping
    # before any encoder call; no model outputs are mocked or manufactured.
    area = contract['height'] * contract['width']
    resized = []
    for image in images:
        width = max(32, round(math.sqrt(area * image.width / image.height) / 32) * 32)
        height = max(32, round(math.sqrt(area * image.height / image.width) / 32) * 32)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append(image)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, 4)
        pixels = pixels[None].permute(0, 3, 1, 2).unsqueeze(2).float() / 255.0
        vae_pixels.append((2 * pixels - 1).to(torch.bfloat16))
    context = SimpleNamespace(processor=capture)
    def prompt_call(prompt):
        QwenImage21EncodingStage._encode_prompt(context, prompt, resized, torch.device('cpu'))
    batch = SimpleNamespace(height=contract['height'], width=contract['width'])
    shape = config.prepare_latent_shape(batch, 1, 1)
    dtype = config.get_latent_dtype(torch.bfloat16)
    noise = randn_tensor(shape, generator=generator, device='cpu', dtype=dtype)
    noise = config.maybe_pack_latents(noise, 1, batch)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model / 'scheduler', local_files_only=True)
    if hasattr(scheduler, 'init_noise_sigma'):
        noise = noise * scheduler.init_noise_sigma
    sched = scheduler.config
    mu = calculate_linear_shift(noise.shape[1], base_seq_len=sched.get('base_image_seq_len',256),
        max_seq_len=sched.get('max_image_seq_len',4096), base_shift=sched.get('base_shift',.5),
        max_shift=sched.get('max_shift',1.15))
    scheduler.set_timesteps(sigmas=config.prepare_sigmas(None, steps), device='cpu', mu=mu)
    objects = [QwenImage21EncodingStage, QwenImage21PipelineConfig, FlowMatchEulerDiscreteScheduler]
else:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline, get_qwen_image_21_pre_process_func
    preprocess = get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(model), extras=model_index))
    request = SimpleNamespace(prompt={'prompt':contract['prompts'][0]['prompt'],
        'multi_modal_data':{'image':images} if images else {}},
        sampling_params=SimpleNamespace(true_cfg_scale=contract['true_cfg_scale'], sigmas=None,
            extra_args={}, height=contract['height'], width=contract['width']))
    request = preprocess(request)
    additional = request.prompt['additional_information']
    resized = additional.get('prompt_image', [])
    vae_pixels = [pixels.to(torch.bfloat16) for pixels in additional.get('vae_images', [])]
    system = 'Comprehend and analyze the provided prompt.'
    prefix = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n'
    suffix = '{}<|im_end|>\n<|im_start|>assistant\n'
    # These initializer-only strings/metadata match the frozen native source;
    # constructing the full pipeline would load GPU weights, outside this audit.
    context = SimpleNamespace(processor=capture, text_encoder=SimpleNamespace(dtype=torch.bfloat16),
        prompt_template_t2i=prefix+suffix,
        prompt_template_ti2i=prefix+'<image1><|vision_start|><|image_pad|><|vision_end|>'+suffix,
        _max_length=8192, device=torch.device('cpu'), vae_scale_factor=16, latent_channels=64,
        _pack_latents=QwenImage21Pipeline._pack_latents, sample_sigmas=model_index.get('sample_sigmas'))
    def prompt_call(prompt):
        QwenImage21Pipeline._get_qwen_prompt_embeds(context, prompt, [resized] if resized else None)
    noise, _ = QwenImage21Pipeline.prepare_latents(context, None, 1, 64,
        contract['height'], contract['width'], torch.bfloat16, torch.device('cpu'), generator)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model / 'scheduler', local_files_only=True)
    context.scheduler = scheduler
    QwenImage21Pipeline.prepare_timesteps(context, steps, None, noise.shape[1])
    objects = [QwenImage21Pipeline, get_qwen_image_21_pre_process_func, FlowMatchEulerDiscreteScheduler]

rows = []
for prompt in contract['prompts']:
    capture.last = None
    try:
        prompt_call(prompt['prompt'])
    except ProcessorCaptured:
        pass
    else:
        raise AssertionError('Native method did not reach processor boundary')
    assert capture.last is not None
    rows.append(dict(id=prompt['id'], **capture.last))
for obj in objects:
    path = Path(inspect.getsourcefile(obj))
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(arm=args.arm, cpu_preflight_only=True,
    contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
    reference_sha256=reference_sha, prompts=rows,
    vae_input_pixels=[dict(**signature(p), stride=list(p.stride())) for p in vae_pixels],
    initial_packed_noise=signature(noise), timesteps=signature(scheduler.timesteps),
    sigmas=signature(scheduler.sigmas), sigma_values=scheduler.sigmas.tolist(),
    native_step_count=len(scheduler.timesteps), native_source_sha256=sources,
    vae_sampling_semantics='Both native reference encoders use posterior mode/argmax; encoder weights are not executed in this CPU audit.',
    limitations='Real native prompt methods call the actual checkpoint processor and deliberately stop before encoder execution. Initializer metadata and SGL resize/pixel preparation are source-derived. This verifies CPU conditioning/noise/scheduler inputs, not encoder hidden-state equivalence, DiT image-slot layouts, GPU precision, output quality or performance.')
target = Path('/task/evidence/preflight') / f'{contract["matrix_workload"]}-{args.arm}-inputs.json'
assert not target.exists(), 'Retain earlier CPU audit; select a new evidence label before rerunning'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(artifact=str(target), noise=result['initial_packed_noise'], steps=result['native_step_count'])))
