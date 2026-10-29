"""Archive completed LN candidate measurements and compatibility, excluding weights."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
assert json.loads((root/'sana-video-ln-v1-paired1-status.json').read_text())['status'] == 'raw_complete_audit_pending'
assert json.loads((root/'sana-video-ln-final-compatibility1-status.json').read_text())['status'] == 'raw_complete_quality_replay_and_profile_audit_pending'
paths = []
for relative, patterns in [
    ('outputs/qualification', ['*sana-video-ln-v1-native1*', '*sana-video-ln-final-compatibility1*']),
    ('outputs/benchmarks', ['sana-video-main-eager-aa1-*', 'sana-video-ln-v1-paired1-*']),
    ('outputs/profiles', ['sana-video-ln-v1-qualification1-sgl', 'sana-video-ln-final-compatibility1-*']),
    ('outputs/diagnostics', ['sana-video-ln-final-compatibility1-*', 'sana-video-ln-final-cpu-compile1']),
    ('block-runs', ['sana-video-ln-v1-qualification1-*', 'sana-video-main-eager-aa1-*', 'sana-video-ln-v1-paired1-*',
                    'sana-video-ln-final-compatibility1-*', 'sana-video-ln-final-tests1', 'sana-video-ln-final-tests2']),
    ('evidence/preflight', ['sana-video-ln-*.json', 'sana-video-main-eager-aa1-admission.json']),
    ('', ['sana-video-ln-*-status.json', 'sana-video-ln-*-queue.log', '*sana_video*1.py', '*sana_video*2.py', '*sana_video*1.sh',
          'sana-video-ln-*-plan*.json', 'sana-video-ln-v1-paired1.json', 'sana-video-final-quality-sequence1.json']),
]:
    for pattern in patterns:
        for item in (root/relative).glob(pattern):
            paths.extend([item] if item.is_file() else [p for p in item.rglob('*') if p.is_file()])
paths = sorted(set(paths))
assert paths and all('cache' not in p.relative_to(root).parts and not p.is_symlink() for p in paths)
assert any('sana-video-ln-final-tests1/numerical.log' in str(p) for p in paths)
assert sum(p.name == 'output.mp4' for p in paths) == 5+16+48+51
name = 'sana-video-ln-qualification1-archive'
manifest = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
archive = root/(name+'.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz', compresslevel=3) as stream:
    for path in paths:
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=manifest)
(root/(name+'-manifest.json')).write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key != 'members'}), len(manifest), flush=True)
