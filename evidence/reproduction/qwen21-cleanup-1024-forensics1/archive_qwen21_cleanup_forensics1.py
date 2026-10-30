"""Archive the separate failed-cleanup investigation, retaining the failed primary receipt."""
import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()
assert args.run == 'qwen21-cleanup-1024-forensics1'
plan_path, status_path = root/(args.run+'.json'), root/(args.run+'-status.json')
plan, status = json.loads(plan_path.read_text()), json.loads(status_path.read_text())
assert status['status'] == 'raw_complete_audit_pending'
assert len(status['completed']) == plan['expected_processes']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
paths = {plan_path,status_path,Path(__file__).resolve()}
for area,pattern in [('block-runs',args.run+'-*'),('outputs/benchmarks',args.run+'-*'),
                     ('outputs/profiles',args.run+'-*'),('outputs/diagnostics',args.run+'-*'),
                     ('outputs/qualification','sgl-qwen21-base-*-'+args.run+'-*')]:
    for item in (root/area).glob(pattern):
        paths.update([item] if item.is_file() else (p for p in item.rglob('*') if p.is_file()))
for item in plan.get('files',[]):
    paths.add(root/item['path'])
for path in plan.get('sha256',{}):
    paths.add(root/(Path(path).name if path.startswith('scripts/') else path))
paths.update(root/path for path in plan['required_admissions'])
paths.update(p for p in root.glob(args.run+'*.log') if p.is_file())
assert sum(p.name == 'output.png' for p in paths) == plan['expected_outputs']
assert sum(p.name.endswith('.trace.json.gz') for p in paths) == plan['expected_traces']
assert all(p.is_file() and not p.is_symlink() and 'cache' not in p.relative_to(root).parts for p in paths)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


members = [dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)]
archive = root/(args.run+'-archive.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
    for path in sorted(paths):
        stream.add(path,arcname=str(path.relative_to(root)),recursive=False)
result = dict(archive=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,members=members,
              generated_images=plan['expected_outputs'],traces=plan['expected_traces'],local_audit_pending=True)
manifest = root/(args.run+'-archive-manifest.json')
assert not manifest.exists()
manifest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key != 'members'}))
