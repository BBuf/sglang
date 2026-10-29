"""Publishable evidence export; retain original archives and every negative result."""
import hashlib
import json
import re
import shutil
from pathlib import Path

R=Path(__file__).resolve().parents[1]
E=R/'evidence'
D=R/'artifacts/anima-rope-rounding'
def read(path):return json.loads(path.read_text())
assert read(E/'preflight/anima-rope-final-cleanup-admission.json')['final_model_gain_remains']
assert read(E/'preflight/anima-final-bcg-audit2-audit.json')['status']=='pass'
assert read(E/'profiles/anima-rope-final-profile-audit.json')['all_final_dispatch_pass']
assert not read(E/'media/anima-rope-final/manifest.json')['visual_review_pending']
for name in ['anima-initial-evidence1','anima-backend-screens1','anima-final-evidence1','anima-final-followup2']:
    receipt=read(E/'checkpoints'/f'{name}-local-verification.json')
    assert receipt.get('all_member_hashes_verified') or receipt.get('all_member_hashes_equal'), receipt
D.mkdir(parents=True,exist_ok=True)
image_map={}
omitted=[]
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
def export(source):
    relative=source.relative_to(R)
    target=D/relative
    if source.suffix=='.png':
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        dedup=D/'images/by-sha256'/(digest+'.png')
        if not dedup.exists():copy(source,dedup)
        image_map[str(relative)]=str(dedup.relative_to(D))
    elif source.name=='response.json':
        data=read(source)
        assert 'data' in data
        data.pop('data')
        data['_public_export']=dict(original_response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),omitted='Duplicate base64 PNG. Every unique PNG is exported by hash; image-path-map.json resolves original paths. Full original responses remain in verified local archives.')
        target=target.with_name('response-metrics.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(data,indent=2)+'\n')
    elif source.suffix in ['.pt','.bundle']:
        omitted.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reason='Binary tensor snapshots or git transfer bundle remain in local archives; audit statistics/source hashes and reproduction scripts are public.'))
    else:
        assert source.stat().st_size<100*1024*1024, source
        copy(source,target)

for area in ['qualification','preflight','diagnostics','profiles','block-runs','benchmarks','history','plans','runner-logs','downloads','harness-versions']:
    for path in (E/area).glob('*anima*'):
        for source in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if source.is_file():export(source)
for source in (E/'media/anima-rope-final').iterdir():
    if source.is_file():copy(source,D/'media'/source.name)
for source in (R/'scripts').glob('*anima*'):
    if source.suffix in ['.py','.sh']:copy(source,D/'reproduce'/source.name)
for name in ['compare_moss_qualification.py','summarize_moss_trace.py','diffusion_skill_env.py','preflight_ming_environment.py']:
    path=R/'scripts'/name
    if path.exists():copy(path,D/'reproduce'/name)
for source in (R/'inputs').glob('anima*'):copy(source,D/'inputs'/source.name)
for source in (E/'checkpoints').glob('anima*.json'):copy(source,D/'archive-manifests'/source.name)
for name in ['cleanup-style-notes.md']:
    copy(E/'history'/name,D/'evidence/history'/name)
copy(R/'matrix-draft.json',D/'frozen-matrix-with-execution.json')
copy(E/'anima-rope-final-performance-table.md',D/'performance-table.md')
copy(E/'preflight/hardware.txt',D/'hardware.txt')
for arm,filename in [('sgl','sglang-pip-freeze-final.txt'),('omni','omni-pip-freeze-final.txt')]:
    pins=[line for line in (E/'preflight/container-logs'/filename).read_text().splitlines() if re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line)]
    (D/f'{arm}-version-pins.txt').write_text('\n'.join(pins)+'\n')
(D/'image-path-map.json').write_text(json.dumps(image_map,indent=2)+'\n')
(D/'omitted-binary-files.json').write_text(json.dumps(omitted,indent=2)+'\n')
manifest={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in D.rglob('*') if p.is_file() and '.git' not in p.parts and p.name!='evidence-files-sha256.json'}
assert all(row['bytes']<100*1024*1024 for row in manifest.values())
(D/'evidence-files-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),image_paths=len(image_map),unique_images=len(set(image_map.values())),omitted_binaries=len(omitted))))
