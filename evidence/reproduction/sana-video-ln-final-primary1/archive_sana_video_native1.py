"""Archive completed native video qualification and observer evidence, excluding weights."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
assert json.loads((root/'sana-video-native-qualification1-status.json').read_text())['active'] is None
assert json.loads((root/'sana-video-tensor-audit1-status.json').read_text())['status'] == 'observed_native_outputs_exact_runtime_tensor_audit_pending'
paths = []
for relative, patterns in [
    ('outputs/qualification', ['*sana-video*t2v-native1*', '*sana-video*t2v-audit1*']),
    ('outputs/profiles', ['sana-video-native-qualification1-sgl']),
    ('outputs/diagnostics', ['sana-video-tensor-audit1', 'sana-video-noise-schedule1']),
    ('block-runs', ['sana-video-native-qualification1-*', 'sana-video-tensor-audit1-*']),
]:
    for pattern in patterns:
        for item in (root/relative).glob(pattern):
            paths.extend([item] if item.is_file() else [p for p in item.rglob('*') if p.is_file()])
for name in ['sana-video-noise-schedule1.json', 'sana-video-runtime-input-audit1.json', 'sana-video-native-config1.json']:
    path = root/'evidence/preflight'/name
    if path.exists():
        paths.append(path)
paths = sorted(set(paths))
assert paths and all('cache' not in p.relative_to(root).parts and not p.is_symlink() for p in paths)
name = 'sana-video-native1-archive'
manifest = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
archive = root/(name+'.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz', compresslevel=3) as stream:
    for path in paths:
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=manifest)
(root/(name+'-manifest.json')).write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='members'}), len(manifest), flush=True)
