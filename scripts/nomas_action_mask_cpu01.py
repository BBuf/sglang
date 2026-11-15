"""Compare native action visibility with one spatial token per latent frame."""
import datetime
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

root=Path('/task');arm=sys.argv[1];assert arm in ('sgl','omni')
torch.set_num_threads(2)
source=root/f'evidence/preflight/nomas-cpu-contract01-{arm}.json'
contract=json.loads(source.read_text());out=root/f'evidence/preflight/nomas-action-mask-cpu01-{arm}.json';assert not out.exists()
if arm=='sgl':
    from sglang.multimodal_gen.runtime.layers.kvcache.nomas_segment_cache import NomasAttentionPlan
    implementation=Path(inspect.getfile(NomasAttentionPlan))
else:
    from vllm_omni.diffusion.models.hy_nomas.attention import action_ranges
    implementation=Path(inspect.getfile(action_ranges))
rows=[]
for item in contract['requests']:
    text=torch.tensor(item['cropped_attention_mask'],dtype=torch.bool)
    history=torch.zeros(1,0,dtype=torch.bool)
    frames=item['latent_shape'][2];start,offsets,windows=item['actions']
    valid=torch.cat((torch.ones(1,frames,dtype=torch.bool),text),dim=1)
    n=valid.shape[1]
    if arm=='sgl':
        plan=NomasAttentionPlan(text,history,frames,1,offsets,windows,backend='sdpa')
        mask=plan.mask & plan.query_valid[0,:,None]
    else:
        rectangles=action_ranges(history,text,frames,[offsets],windows,1)[0]
        mask=torch.zeros(n,n,dtype=torch.bool)
        for q0,q1,k0,k1 in rectangles:mask[q0:q1,k0:k1]=True
        mask &= valid[0,:,None] & valid[0,None,:]
    rows.append(dict(id=item['id'],shape=list(mask.shape),visible_pairs=int(mask.sum()),mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest()))
assert not torch.cuda.is_initialized()
j=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,benchmark=False,
       scope='Native CPU SDPA reference mask versus native Omni action rectangles, both exclude padded queries/keys; all21 frozen prompts and latent frame counts, spatial_tokens=1. This checks temporal/text visibility, not actual CUDA mask/kernels or model quality.',
       contract_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),implementation=str(implementation),implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),requests=rows)
out.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(dict(arm=arm,requests=len(rows),cuda_initialized=torch.cuda.is_initialized())))
