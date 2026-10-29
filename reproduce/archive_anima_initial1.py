"""Archive only completed Anima evidence, with per-file hashes; no checkpoints."""
import hashlib,json,tarfile
from pathlib import Path
R=Path('/data/xyzhang/diffusion-rtx6000-20261026')
paths=[]
for relative,pattern in [
 ('outputs/qualification','*anima*t2i-native1*'),
 ('outputs/qualification','*anima*audit1*'),
 ('outputs/qualification','*anima*rope-v1*'),
 ('outputs/diagnostics','anima-tensor-audit1'),
 ('outputs/diagnostics','anima-rope-numerics2'),
 ('outputs/profiles','anima-native-qualification1-*'),
 ('outputs/profiles','anima-rope-v1-qualification1-*'),
 ('block-runs','anima-native-qualification1-*'),
 ('block-runs','anima-tensor-audit1-*'),
 ('block-runs','anima-rope-v1-qualification1-*'),
]:
 for item in (R/relative).glob(pattern):
  paths.extend([item] if item.is_file() else [p for p in item.rglob('*') if p.is_file()])
paths=sorted(set(paths))
assert paths and all('cache' not in p.relative_to(R).parts for p in paths)
name='anima-initial-evidence1-archive'
manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
archive=R/(name+'.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
 for p in paths:tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
receipt=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=manifest)
(R/(name+'-manifest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'}),len(manifest),flush=True)
