"""Verify original native Omni checkpoint and SGL component tensor contents."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

torch.set_num_threads(4)
root=Path('/task')
out=root/'evidence/preflight/anima-checkpoint-equivalence1.json'
assert not out.exists()
for label in ['anima','anima-components']:
    manifest=json.loads((root/f'evidence/downloads/{label}-modelscope-download-manifest.json').read_text())
    assert manifest['complete'] and all(r['verified_against_hf'] for r in manifest['files'])
original=root/'cache/common/Anima-Base-v1.0-official/split_files/diffusion_models/anima-base-v1.0.safetensors'
components=root/'cache/common/Anima-Base-v1.0-Diffusers'
records=[]
with safe_open(original,framework='pt',device='cpu') as source:
    # Use native conversion functions on key placeholders, then compare bytes
    # from original tensors with their declared converted component tensors.
    keys={k:k for k in source.keys()}
    transformer,conditioner=AnimaPipeline._split_single_file_state_dict(keys)
    transformer=AnimaPipeline._convert_original_transformer_state_dict(transformer)
    for name,mapping in [('transformer',transformer),('text_conditioner',conditioner)]:
        with safe_open(components/name/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as target:
            source_keys=set(mapping);target_keys=set(target.keys())
            assert source_keys==target_keys,(name,sorted(source_keys-target_keys)[:15],sorted(target_keys-source_keys)[:15])
            for key in sorted(source_keys):
                a,b=source.get_tensor(mapping[key]),target.get_tensor(key)
                exact=a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
                records.append(dict(component=name,key=key,original_key=mapping[key],shape=list(a.shape),
                                    source_dtype=str(a.dtype),target_dtype=str(b.dtype),exact=exact))
result=dict(all_tensors_equal=all(r['exact'] for r in records),tensor_count=len(records),records=records,
            original=str(original),components=str(components),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
assert result['all_tensors_equal']
