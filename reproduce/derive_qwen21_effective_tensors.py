"""Compare equivalent boundaries from archived observation-only tensors, CPU only."""
import json
import hashlib
from pathlib import Path
import torch
ROOT=Path('/task')
L=ROOT/'outputs/diagnostics/qwen21-numerics1-sgl';R=ROOT/'outputs/diagnostics/qwen21-numerics1-omni'
torch.set_num_threads(8)
def metric(a,b):
 e=(a.double()-b.double()).abs()
 return dict(shape=list(a.shape),exact=torch.equal(a,b),max_abs=float(e.max()),relative_l2=float(torch.linalg.vector_norm(e)/torch.linalg.vector_norm(a.double()).clamp_min(1e-30)))
rows=[]
# Resolve the token from the verified checkpoint rather than assuming its numeric ID.
tokenizer=json.loads((ROOT/'cache/common/Qwen-Image-2.1/processor/tokenizer.json').read_text())
image_id=next(t['id'] for t in tokenizer['added_tokens'] if t['content']=='<|image_pad|>')
for p in sorted(L.iterdir()):
 meta=json.loads((p/'metadata.json').read_text());peer=R/p.name
 row=dict(key=p.name,prompt=meta['prompt'],effective_model_timesteps=[])
 for f in sorted(p.glob('step-*-before.pt')):
  a=torch.load(f,weights_only=True);b=torch.load(peer/f.name,weights_only=True)
  effective=a['model_timestep'].to(a['sample'].dtype)/1000
  row['effective_model_timesteps'].append(dict(file=f.name,sgl_raw=a['model_timestep'].tolist(),omni_raw=b['model_timestep'].tolist(),comparison=metric(effective,b['model_timestep'])))
 if (p/'text_encoder.pt').exists() and (peer/'text_encoder.pt').exists():
  a=torch.load(p/'text_encoder.pt',weights_only=True);b=torch.load(peer/'text_encoder.pt',weights_only=True);assert torch.equal(a['ids'],b['ids'])
  mask=a['ids']==image_id
  row['image_tokens']=int(mask.sum());row['non_image_hidden']=metric(a['hidden'][~mask],b['hidden'][~mask])
  if mask.any():row['image_hidden']=metric(a['hidden'][mask],b['hidden'][mask])
 rows.append(row)
out=ROOT/'evidence/preflight/qwen21-native-stage-equivalent-boundaries.json';assert not out.exists()
out.write_text(json.dumps(dict(diagnostic_only=True,image_token_id=image_id,formula='SGL transformer forward uses timestep.to(images.dtype)/1000; Omni capture already normalized BF16.',raw_comparison='qwen21-native-stage-tensor-comparison.json retained unchanged; raw 999 difference is different capture boundary, not an error.',rows=rows),indent=2)+'\n')
for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='prompt'}))
