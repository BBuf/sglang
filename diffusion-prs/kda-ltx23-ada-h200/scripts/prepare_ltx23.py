#!/usr/bin/env python3
"""Use the exact native LTX2.3 materializer in an owned cache, protect shared metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parent
owned=root/'model-cache/ltx23'
if args.resume:
 assert (owned/'.campaign-owned').is_file(), 'Refusing to reuse unmarked cache'
else:
 owned.mkdir(parents=True,exist_ok=False)
(owned/'.campaign-owned').write_text('KDA H200 native LTX2.3 materialization, removable after validation.\n')
hub=owned/'hub';hub.mkdir(exist_ok=args.resume)
shared=Path('/cluster-storage/models')
revisions={
 'Lightricks/LTX-2':'47da56e2ad66ce4125a9922b4a8826bf407f9d0a',
 'Lightricks/LTX-2.3':'7caa482d5cd10a2eae6b34cb48f093ebc45a263e',
 'FastVideo/LTX-2.3-Distilled-Diffusers':'22b09fb1860a944bf10fa21f033d957d9ab9ec20',
 'MickJ/LTX-2.3-overlay':'e0cc94f279ec16bb87c230134d40319f6ce40c5e',
}
shared_metadata={}
for repo,revision in revisions.items():
 name='models--'+repo.replace('/','--');source=shared/name/'snapshots'/revision;dest=hub/name/'snapshots'/revision
 assert source.is_dir(),source
 for path in source.rglob('*'):
  relative=path.relative_to(source)
  if '__pycache__' in relative.parts:continue
  target=dest/relative
  if path.is_dir():target.mkdir(parents=True,exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True)
  if path.suffix.lower() in {'.safetensors','.bin','.pth','.pt'}:
   if not target.exists():target.symlink_to(path.resolve())
  else:
   shutil.copy2(path,target)
   if path.suffix.lower()=='.json':shared_metadata[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 (hub/name/'refs').mkdir(exist_ok=True)
 (hub/name/'refs/main').write_text(revision)
(owned/'shared-metadata-before.json').write_text(json.dumps(shared_metadata,indent=2))
os.environ.update(HF_HOME=str(owned/'hf-home'),HF_HUB_CACHE=str(hub),HUGGINGFACE_HUB_CACHE=str(hub),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='1',SGLANG_DIFFUSION_CACHE_ROOT=str(owned/'native-cache'),FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0,str(root.parent/'baseline/python'))
from huggingface_hub import hf_hub_download
print('Downloading only missing native-required distilled LoRA asset (7,605,507,256 bytes); checkpoint and donors are linked from shared cache.',flush=True)
asset=hf_hub_download(repo_id='Lightricks/LTX-2.3',filename='ltx-2.3-22b-distilled-lora-384.safetensors',revision=revisions['Lightricks/LTX-2.3'],cache_dir=str(hub))
print('Downloaded',asset,flush=True)
# Match native CLI import ordering before importing the overlay utilities.
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.utils.model_overlay import materialize_overlay_model
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import _verify_diffusers_model_complete
source=hub/'models--Lightricks--LTX-2.3/snapshots'/revisions['Lightricks/LTX-2.3']
overlay=hub/'models--MickJ--LTX-2.3-overlay/snapshots'/revisions['MickJ/LTX-2.3-overlay']
result=materialize_overlay_model(source_model_id='Lightricks/LTX-2.3',overlay_spec={'overlay_repo_id':'MickJ/LTX-2.3-overlay','overlay_revision':revisions['MickJ/LTX-2.3-overlay']},overlay_dir=str(overlay),source_dir=str(source),verify_diffusers_model_complete_fn=_verify_diffusers_model_complete)
assert _verify_diffusers_model_complete(result)
config=json.loads((Path(result)/'transformer/config.json').read_text())
assert config['cross_attention_adaln'] is True
shared_after={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in shared_metadata}
assert shared_after==shared_metadata,'Shared metadata changed unexpectedly'
record=dict(model='Lightricks/LTX-2.3',revisions=revisions,local_model_path=result,cross_attention_adaln=config['cross_attention_adaln'],shared_metadata_unchanged=True,shared_metadata=shared_after,owned_cache=str(owned),downloaded_asset_bytes=Path(asset).stat().st_size,note='Native overlay materializer unchanged; only cache metadata was physically copied to isolate its writes. All weights/other native donors are exact pinned files.')
(root/'ltx23-prepared.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='shared_metadata'},indent=2),flush=True)
