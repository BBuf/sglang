"""Call native token/action/segment/scheduler code without loading model weights."""
import datetime
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from transformers import AutoTokenizer

root = Path('/task')
arm = sys.argv[1]
assert arm in ('sgl','omni')
assert not torch.cuda.is_initialized()
manifest = root/'inputs/nomas-native-workloads01.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='0d407ba41a9749f65c9f29e131ec35a2665ab9f3e73d3896b2351b28041aff7e'
workloads = json.loads(manifest.read_text())
tokenizer = AutoTokenizer.from_pretrained(root/'inputs/nomas-metadata01/Hy-Nomas-Diffusers/tokenizer',padding_side='right',local_files_only=True)
out = root/'evidence/preflight'/f'nomas-cpu-contract01-{arm}.json'
assert not out.exists()
sources = {}
def source(obj):
    path = Path(inspect.getfile(obj))
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

if arm=='sgl':
    from diffusers import FlowMatchEulerDiscreteScheduler
    from sglang.multimodal_gen.configs.pipeline_configs.nomas import NomasPipelineConfig
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import action_metadata,canonical_caption,plan_segments
    config = NomasPipelineConfig()
    for obj in (NomasPipelineConfig,action_metadata,FlowMatchEulerDiscreteScheduler):source(obj)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root/'inputs/nomas-metadata01/Hy-Nomas-Diffusers/scheduler',local_files_only=True)
    scheduler.set_timesteps(num_inference_steps=8,device='cpu',sigmas=config.prepare_sigmas(None,8))
else:
    from vllm_omni.diffusion.models.hy_nomas.text_encoder import TextEncoder
    from vllm_omni.diffusion.models.hy_nomas.scheduler import EulerSolver
    from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import action_conditioning,normalize_request,plan_chunks
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    for obj in (TextEncoder,EulerSolver,action_conditioning,OmniDiffusionSamplingParams):source(obj)
    # Run the actual native token method with only its tokenizer state;
    # this intentionally does not claim a loaded or executed text encoder.
    encoder = SimpleNamespace(tokenizer=tokenizer,prompt_template_video={'crop_start':-1})
    encoder.text2tokens = MethodType(TextEncoder.text2tokens,encoder)
    encoder.text2tokens('a photo of a cat',data_type='video')
    scheduler = EulerSolver()
    scheduler.set_timesteps(8,device='cpu')

rows=[]
for item in workloads['requests']:
    text = workloads['prompts'][item['prompt_id']]
    assert hashlib.sha256(text.encode()).hexdigest()==workloads['prompt_sha256'][item['prompt_id']]
    frames = (item['frames']-1)//4+1
    if arm=='sgl':
        canonical = canonical_caption(text,'json')
        assert canonical==text, 'Native canonicalization changed the frozen string'
        tokens = config.tokenize_prompt([text],tokenizer,{'max_length':1280})
        crop = int(tokens['nomas_crop_start'])
        mask = tokens['attention_mask'][:,crop:]
        action = action_metadata(text,config,tokenizer,mask,frames,item['fps'])
        plan = plan_segments(item['frames'],text,[{'frames':item['frames'],'caption':text}],None,config.segment_latent_frames)
        segments = [[p.start,p.end] for p in plan]
        shape = config.prepare_latent_shape(SimpleNamespace(height=item['height'],width=item['width'],guidance_scale=1.0),1,frames)
    else:
        tokens = encoder.text2tokens(text,data_type='video',max_length=1280)
        crop = encoder.prompt_template_video['crop_start']
        mask = tokens['attention_mask'][:,crop:]
        action = action_conditioning(text,encoder,1280,frames)
        params = OmniDiffusionSamplingParams(num_frames=item['frames'],height=item['height'],width=item['width'],num_inference_steps=8,
                                             guidance_scale=1.0,seed=item['seed'],fps=24.0,extra_args={'flow_shift':5.0})
        request = normalize_request(text,params)
        plan = plan_chunks(request,frames)
        segments = [[p.start,p.end] for p in plan]
        shape = (1,32,frames,request.height//32,request.width//32)
    rows.append(dict(id=item['id'],prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),crop=crop,
                     input_ids=tokens['input_ids'].tolist(),attention_mask=tokens['attention_mask'].tolist(),
                     cropped_attention_mask=mask.tolist(),actions=action,segments=segments,latent_shape=shape))
assert not torch.cuda.is_initialized()
report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,benchmark=False,
            scope='CPU native tokenization, action/segment preparation and scheduler only; no weights, native encoder/DiT execution or generated quality claim.',
            torch=torch.__version__,source_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timesteps=scheduler.timesteps.tolist(),sigmas=scheduler.sigmas.tolist(),requests=rows)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(arm=arm,requests=len(rows),crops=sorted({x['crop'] for x in rows}),sigmas=report['sigmas'])))
