"""Export complete numerical evidence and deduplicated actual images for a failed gate."""
import gzip,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'evidence';D=R/'artifacts/qwen21-current-head-cleanup-failed-188169fa'
read=lambda p:json.loads(p.read_text())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert read(E/'preflight/qwen21-current-head-1024-primary1-admission.json')['passed'] is False
assert read(E/'preflight/qwen21-cleanup-1024-forensics1-audit.json')['all_raw_audits_pass']
media=E/'media/qwen21-current-head-cleanup-failed-188169fa';m=read(media/'manifest.json');review=read(media/'visual-review.json')
assert review['passed'] and review['all_12_panels_inspected'] and review['manifest_sha256']==digest(media/'manifest.json')
runs=[f'qwen21-current-head-{s}-{p}1' for s in [512,1024] for p in ['aa','primary']]+['qwen21-cleanup-1024-forensics1']
selected=set();omitted=[];mapping={};archives=[]
def local_member(run,name):
 p=Path(name)
 if p.parts[0]=='outputs':return E/Path(*p.parts[1:])
 if p.parts[0]=='block-runs':return E/p
 if p.parts[0] in ['evidence','inputs']:return R/p
 if len(p.parts)==1:return E/'reproduction'/run/p
 raise AssertionError(name)
for run in runs:
 archive=read(E/'checkpoints'/(run+'-archive-manifest.json'));v=read(E/'checkpoints'/(run+'-local-verification.json'))
 assert v['all_member_hashes_verified'] and v['all_members_extracted'] and v['sha256']==archive['sha256']
 archives.append(dict(run=run,bytes=archive['bytes'],sha256=archive['sha256'],members=len(archive['members'])))
 omitted.append(dict(path='evidence/checkpoints/'+archive['archive'],bytes=archive['bytes'],sha256=archive['sha256'],reason='Full verified raw archive retained locally; per-member manifests and original numerical evidence exported.'))
 for row in archive['members']:
  p=local_member(run,row['path']);assert digest(p)==row['sha256'],str(p);selected.add(p)
 for path in [E/'plans'/(run+'.json'),E/'checkpoints'/(run+'-archive-manifest.json'),E/'checkpoints'/(run+'-local-verification.json')]:selected.add(path)
 for area in ['preflight','profiles']:
  selected.update(p for p in (E/area).glob(run+'*.json') if p.is_file())
 plan=read(E/'plans'/(run+'.json'))
 for path,sha in plan['sha256'].items():assert digest(R/path)==sha,path;selected.add(R/path)
 refs={ref for tasks in plan.get('quality_references',{}).values() for references in tasks.values() for ref in references}
 for ref in refs:selected.update(p for p in (E/'qualification'/ref).rglob('*') if p.is_file())
for name in ['qwen21-current-head-cleanup-source-diff1.patch','qwen21-current-head-cleanup-source-review1.json','qwen21-current-head-failure-quality1.json','qwen21-current-head-tests1-admission.json']:
 selected.add(E/'preflight'/name)
for name in ['start_qwen21_cleanup_forensics1.py','finish_qwen21_cleanup_forensics1.py','archive_qwen21_cleanup_forensics1.py','audit_qwen21_cleanup_forensics1.py','analyze_qwen21_cleanup_forensics1.py','render_qwen21_current_head_failure_media1.py','write_qwen21_current_head_failure_report1.py','package_qwen21_current_head_failure1.py','analyze_joy_gelu_profiles.py']:
 selected.add(R/'scripts'/name)
for pattern in ['qwen21-modelscope-download-manifest.json','qwen21-current-head-restore1*']:
 selected.update(p for p in (E/'downloads').glob(pattern) if p.is_file() and p.suffix=='.json')
D.mkdir(exist_ok=False)
def copy(p,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for p in sorted(selected):
 assert p.is_file() and not p.is_symlink() and 'cache' not in p.relative_to(R).parts
 name=str(p.relative_to(R));sha=digest(p);target=D/name
 if p.suffix=='.png':
  target=D/'media/by-sha256'/(sha+'.png')
  if not target.exists():copy(p,target)
  mapping[name]=dict(public_path=str(target.relative_to(D)),sha256=sha)
 elif p.name=='response.json':
  value=read(p);assert isinstance(value,dict) and 'data' in value;value.pop('data')
  value['_public_export']=dict(original_sha256=sha,omitted='Repeated base64 PNG data; original complete response remains in verified archive; actual PNG is exported by SHA256.')
  target=target.with_name('response-metrics.json');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2)+'\n')
  omitted.append(dict(path=name,bytes=p.stat().st_size,sha256=sha,public_metrics=str(target.relative_to(D)),reason=value['_public_export']['omitted']))
 elif p.suffix=='.log' and p.stat().st_size>256*1024:
  target=target.with_name(target.name+'.gz');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
  mapping[name]=dict(public_path=str(target.relative_to(D)),sha256=sha,encoding='gzip')
 else:
  assert p.stat().st_size<100*1024**2 and p.suffix not in ['.pt','.bundle'],name
  copy(p,target)
for p in media.iterdir():
 if p.is_file():copy(p,D/'media'/p.name)
for name in ['hardware.txt','container-logs/sglang-pip-freeze-final.txt','container-logs/omni-pip-freeze-final.txt']:copy(E/'preflight'/name,D/'environment'/Path(name).name)
copy(E/'qwen21-current-head-failure-report1.md',D/'README.md')
for name,data in [('original-path-map.json',mapping),('omitted-content.json',omitted),('complete-raw-archives.json',archives)]:
 (D/name).write_text(json.dumps(data,indent=2)+'\n')
files={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in D.rglob('*') if p.is_file()}
(D/'evidence-files-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(dict(directory=str(D),files=len(files),bytes=sum(x['bytes'] for x in files.values()),raw_archives=archives,unique_source_pngs=len(list((D/'media/by-sha256').glob('*.png'))),panels=len(m['rows']),publication_pending=True)))
