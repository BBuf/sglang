"""Archive all depthwise experiments and completed native evidence, including failures."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
assert json.loads((root/'sana-video-depthwise-v1-paired1-status.json').read_text())['status']=='raw_complete_audit_pending'
assert json.loads((root/'sana-video-depthwise-final-compatibility1-status.json').read_text())['status']=='raw_complete_quality_replay_and_profile_audit_pending'
assert json.loads((root/'evidence/preflight/sana-video-depthwise-final-compatibility1-raw-audit.json').read_text())['passed']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
paths = set()
for area, patterns in [
    ('outputs/qualification',['*sana-video-depthwise*']),
    ('outputs/benchmarks',['sana-video-depthwise*']),
    ('outputs/profiles',['sana-video-depthwise*']),
    ('outputs/diagnostics',['sana-video-depthwise*']),
    ('block-runs',['sana-video-depthwise*']),
    ('evidence/preflight',['*depthwise*']),
    ('sana-video-depthwise-bcg-observer1',['sitecustomize.py']),
    ('',['*depthwise*.json','*depthwise*.log','*depthwise*.py','*depthwise*.sh',
         'run_sana_video_blocks1.py','bench_sana_video_serving1.py','submit_sana_video_compatibility1.py',
         'submit_sana_video_candidate_qualification.py','sana-video-final-quality-sequence1.json']),
]:
    for pattern in patterns:
        for item in (root/area).glob(pattern):
            paths.update([item] if item.is_file() else [p for p in item.rglob('*') if p.is_file()])
paths.add(root/'inputs/sana-video-480p-t2v-workloads.json')
paths = sorted(paths)
assert paths and all('cache' not in p.relative_to(root).parts and not p.is_symlink() for p in paths)
assert sum(p.name=='output.mp4' for p in paths)==5+16+48+51
assert any(p.name=='sana-video-depthwise-v1-qualification1-status.json' for p in paths), 'Initial preparation failure must remain'
assert any('sana-video-depthwise-glu-screen1' in str(p) for p in paths), 'Slower naive screen must remain'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

name = 'sana-video-depthwise-qualification1-archive'
archive = root/(name+'.tar.gz')
assert not archive.exists()
members = [dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
    for path in paths:stream.add(path,arcname=str(path.relative_to(root)),recursive=False)
result = dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),members=members)
(root/(name+'-manifest.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='members'}),len(members),flush=True)
