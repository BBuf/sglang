"""Install the immutable diagnostics only after both complete repair primary admissions."""
import base64
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
receipt=root/'evidence/preflight/joy-input-lifetime-repair1-diagnostics-deployment1.json'
assert not receipt.exists(),'Inspect a previous completed deployment; never duplicate it'
manifest_path=root/'evidence/checkpoints/joy-input-lifetime-repair1-diagnostics-deployment1-manifest.json'
manifest=json.loads(manifest_path.read_text())
payload_path=root/'evidence/checkpoints'/manifest['archive']
payload=payload_path.read_bytes()
assert hashlib.sha256(payload).hexdigest()==manifest['sha256']=='348dac1969727de778a6bd9a868a95f80abe4c09499fb892887959b02ac0afac'
assert manifest['all_members_read_back_verified']
plan=json.loads((root/'evidence/plans/joy-input-lifetime-repair1-diagnostics1.json').read_text())
head='03368cf1c325e526184b8fb223443e1ae931b13c'
assert plan['sources']['current']['head']==head
receipts={}
for name in plan['required_admissions']:
    path=root/name;value=json.loads(path.read_text())
    assert value['passed'] and value['source_sha']==head,name
    receipts[name]=path.read_text()
for shape in ['square','wide']:
    run='joy-input-lifetime-repair1-'+shape+'-primary1'
    path=root/'evidence/checkpoints'/(run+'-local-verification.json')
    verification=json.loads(path.read_text())
    assert verification['all_member_hashes_verified'] and verification['all_members_extracted']
    receipts[str(path.relative_to(root))]=path.read_text()
    path=root/'evidence/benchmarks'/(run+'-analysis.json')
    analysis=json.loads(path.read_text())
    assert analysis['source_and_ownership_audit_pass'] and analysis['all_planned_blocks_present']
    assert len(analysis['blocks'])==12
    receipts[str(path.relative_to(root))]=path.read_text()
code=r'''
import base64,datetime,hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path('/task');packet=json.load(sys.stdin);manifest=packet['manifest'];receipts=packet['receipts'];payload=base64.b64decode(packet['archive_base64'],validate=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as stream:
    members=stream.getmembers();assert all(item.isfile() for item in members)
    assert len(members)==len(manifest['files']) and {p.name for p in members}==set(manifest['files'])
    files={p.name:stream.extractfile(p).read() for p in members}
for name,raw in files.items():
    item=manifest['files'][name];assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
files.update({name:raw.encode() for name,raw in receipts.items()})
for name,raw in files.items():
    rel=pathlib.Path(name);assert not rel.is_absolute() and '..' not in rel.parts
    path=root/rel;assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
    if path.exists():assert path.is_file() and path.read_bytes()==raw,name
plan=json.loads(files['joy-input-lifetime-repair1-diagnostics1.json'])
assert not (root/(plan['id']+'-status.json')).exists()
for name in plan['required_admissions']:
    value=json.loads(files[name]);assert value['passed'] and value['source_sha']==plan['sources']['current']['head']
for shape in ['square','wide']:
    run='joy-input-lifetime-repair1-'+shape+'-primary1'
    state=json.loads((root/(run+'-status.json')).read_text())
    assert state['status']=='raw_complete_audit_pending' and len(state['completed'])==12
    local=json.loads(files['evidence/checkpoints/'+run+'-local-verification.json'])
    remote=json.loads((root/(run+'-archive-manifest.json')).read_text())
    assert local['all_member_hashes_verified'] and local['all_members_extracted'] and local['sha256']==remote['sha256']
for source in plan['sources'].values():
    assert subprocess.check_output(['git','-C',source['repo'],'rev-parse','HEAD'],text=True).strip()==source['head']
    assert not subprocess.check_output(['git','-C',source['repo'],'diff','HEAD','--'])
for name,expected in plan['sha256'].items():
    relative=pathlib.Path(name).name if name.startswith('scripts/') else name
    raw=files[relative] if relative in files else (root/relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected,name
installed=[]
for name,raw in files.items():
    p=root/name
    if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(raw);installed.append(name)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),installed=installed,
    payload_sha256=manifest['sha256'],verified_plan_pins=len(plan['sha256']),source_sha=plan['sources']['current']['head'],
    square_and_wide_primary_passed=True,no_GPU_work_started=True)))
'''
command=['ssh','-o','ConnectTimeout=10','SGL-RTX6000-01',shlex.join(['docker','exec','-i','bbuf-diffusion-rtx6000-20261026-sgl','python3','-c',code])]
packet=json.dumps(dict(manifest=manifest,receipts=receipts,archive_base64=base64.b64encode(payload).decode())).encode()
log=root/'evidence/preflight/joy-input-lifetime-repair1-diagnostics-deployment1.log'
assert not log.exists(),'Inspect an interrupted attempt before repeating'
with log.open('xb') as stream:
    result=subprocess.run(command,input=packet,stdout=subprocess.PIPE,stderr=stream,check=True)
record=json.loads(result.stdout)
record['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
record['manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
