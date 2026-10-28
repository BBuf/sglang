"""Audit archived native outputs without altering raw qualification evidence."""
import base64
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];manifest=[]
for arm,variant in [('sgl','cudnn1'),('omni','casts1')]:
 for task in ['t2i','edit']:
  for suffix in (['','-profile'] if arm=='sgl' else ['']):
   folder=ROOT/'evidence/qualification'/f'{arm}-qwen21-base-{task}-{variant}{suffix}'
   meta=json.loads((folder/'metadata.json').read_text());contract=json.loads((folder/'contract.json').read_text());results=json.loads((folder/'results.json').read_text())
   assert sha(folder/'contract.json')==meta['input_sha256']==sha(ROOT/'inputs'/f'qwen21-base-{task}-workloads.json')
   assert meta['source_sha']=={'sgl':'af21e2b989e482cfdce39689126562f2209d53ac','omni':'817d4c17feb702289a2d2b2cbacfd594c262b7a7'}[arm]
   assert meta['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
   assert not meta['benchmark'] and bool(suffix)==meta['profiled']
   assert len(results)==(1 if suffix else 4)
   output_hashes=[]
   for i,result in enumerate(results):
    prompt=contract['prompts'][i%3];d=folder/Path(result['request']).name
    request=json.loads((d/'request.json').read_text());p=request['payload'];response=json.loads((d/'response.json').read_text())
    assert request['endpoint']==('edits' if task=='edit' else 'generations')
    assert p['prompt']==prompt['prompt'] and int(p['seed'])==1101 and int(p['num_inference_steps'])==50
    assert float(p['true_cfg_scale'])==1 and int(p['n'])==1 and p['size']=='1024x1024'
    assert p['generator_device']=='cpu' and p['output_format']=='png' and p['response_format']=='b64_json'
    assert int(p['output_compression'])==100
    if arm=='sgl': assert p['quality']=='lossless'
    if task=='edit': assert request['reference_image']['sha256']==sha(ROOT/request['reference_image']['path'])
    assert len(response['data'])==1 and base64.b64decode(response['data'][0]['b64_json'],validate=True)==(d/'output.png').read_bytes()
    h=sha(d/'output.png');assert h==result['sha256'];output_hashes.append(h)
    with Image.open(d/'output.png') as im:
     assert im.mode=='RGBA' and im.size==(1024,1024)
     extrema=im.getextrema()
    rows.append(dict(arm=arm,variant=variant,task=task,profiled=bool(suffix),prompt=prompt['id'],sha256=h,RGBA_extrema=extrema))
   if not suffix: assert output_hashes[0]==output_hashes[3]
   for p in sorted(folder.rglob('*')):
    if p.is_file():manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
result=dict(status='pass',outputs=rows,manifest=manifest,checks='Frozen request/contract/source; response base64 exact PNG; native RGBA; deterministic repeated short prompt.',encoding_note='These SG cuDNN runs predate equivalence correction and use PNG level9; Omni casts uses level0. Qualification only, no timing comparison.')
out=ROOT/'evidence/qualification/qwen21-followup-raw-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('Audited',len(rows),'outputs;',len(manifest),'files')
