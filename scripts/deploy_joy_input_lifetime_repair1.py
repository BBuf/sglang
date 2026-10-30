"""Install only the pinned local repair and its separate prospective plans."""
import datetime
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
receipt=root/'evidence/preflight/joy-input-lifetime-repair1-deployment.json'
assert not receipt.exists()
files={}
for p in (root/'evidence/plans').glob('joy-input-lifetime-repair1-*.json'):
    files[p.name]=p.read_bytes()
for name in ['admit_joy_input_lifetime_repair1.py','archive_joy_input_lifetime_repair1.py']:
    files[name]=(root/'scripts'/name).read_bytes()
for name in ['evidence/checkpoints/joy-input-lifetime-repair1.bundle','evidence/preflight/joy-input-lifetime-repair1-tests-admission.json','evidence/preflight/joy-input-lifetime-repair1-self-review.json']:
    files[name]=(root/name).read_bytes()
manifest={p:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for p,b in files.items()}
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for name,raw in files.items():
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
payload=stream.getvalue()
out=root/'evidence/checkpoints/joy-input-lifetime-repair1-deployment.tar.gz'
assert not out.exists();out.write_bytes(payload)
remote=r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path('/task'); manifest=json.loads(sys.argv[1]);payload=sys.stdin.buffer.read()
assert hashlib.sha256(payload).hexdigest()==sys.argv[2]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as stream:
    members=stream.getmembers();assert all(p.isfile() for p in members)
    assert len(members)==len(manifest) and {p.name for p in members}==set(manifest)
    raw={p.name:stream.extractfile(p).read() for p in members}
for name,data in raw.items():
    rel=pathlib.Path(name);assert not rel.is_absolute() and '..' not in rel.parts
    p=root/rel;assert p.resolve().is_relative_to(root.resolve()) and not p.exists() and not p.is_symlink()
    assert len(data)==manifest[name]['bytes'] and hashlib.sha256(data).hexdigest()==manifest[name]['sha256']
for name,data in raw.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(data)
repo=root/'repos/sglang';bundle=root/'evidence/checkpoints/joy-input-lifetime-repair1.bundle'
subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],check=True,stdout=sys.stderr)
subprocess.run(['git','-C',str(repo),'fetch','--no-tags',str(bundle),'HEAD'],check=True,stdout=sys.stderr)
head='03368cf1c325e526184b8fb223443e1ae931b13c';worktree=root/'candidates/joy-input-lifetime-repair1'
subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(worktree),head],check=True,stdout=sys.stderr)
assert subprocess.check_output(['git','-C',str(worktree),'rev-parse','HEAD'],text=True).strip()==head
assert not subprocess.check_output(['git','-C',str(worktree),'status','--porcelain'])
a=json.loads((root/'evidence/preflight/joy-input-lifetime-repair1-tests-admission.json').read_text())
assert a['passed'] and a['source_sha']==head
for name,sha in a['source_files_sha256'].items():assert hashlib.sha256((worktree/name).read_bytes()).hexdigest()==sha
for name in ['square-aa1','square-primary1','wide-aa1','wide-primary1']:
    plan=json.loads((root/('joy-input-lifetime-repair1-'+name+'.json')).read_text())
    for p,sha in plan['sha256'].items():
        actual=root/(pathlib.Path(p).name if p.startswith('scripts/') else p)
        assert hashlib.sha256(actual.read_bytes()).hexdigest()==sha,p
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(source_sha=head,clean=True,all_plan_payload_pins_verified=True,files=len(raw),gpu_idle=True,no_GPU_work_started=True)))
'''
args=['ssh','-o','ConnectTimeout=10','SGL-RTX6000-01',shlex.join(['docker','exec','-i','bbuf-diffusion-rtx6000-20261026-sgl','python3','-c',remote,json.dumps(manifest),hashlib.sha256(payload).hexdigest()])]
with (root/'evidence/preflight/joy-input-lifetime-repair1-deployment.log').open('xb') as stream:
    result=subprocess.run(args,input=payload,stdout=subprocess.PIPE,stderr=stream,check=True)
j=json.loads(result.stdout);j.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),payload_sha256=hashlib.sha256(payload).hexdigest(),manifest=manifest)
receipt.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:v for k,v in j.items() if k!='manifest'},indent=2))
