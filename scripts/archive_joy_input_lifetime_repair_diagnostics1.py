"""Archive all current-head diagnostic outputs, including actual test logs and traces."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path('/data/xyzhang/diffusion-rtx6000-20261026')
name = 'joy-input-lifetime-repair1-diagnostics1'
plan_path = root/(name+'.json')
status_path = root/(name+'-status.json')
plan = json.loads(plan_path.read_text())
status = json.loads(status_path.read_text())
assert status['status'] == 'raw_complete_archive_and_local_audit_pending'
assert status['outputs'] == plan['expected_outputs']
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
paths = {plan_path, status_path, Path(__file__).resolve()}
for area, pattern in [('block-runs', name+'-*'), ('outputs/profiles', name+'-*'),
                      ('outputs/qualification', 'sgl-joy-'+name+'-*'),
                      ('outputs/diagnostics', 'joy-input-lifetime-repair1-red-attempt2'),
                      ('outputs/diagnostics', 'joy-input-lifetime-repair1-green-attempt1')]:
    for item in (root/area).glob(pattern):
        paths.update([item] if item.is_file() else (p for p in item.rglob('*') if p.is_file()))
for path in plan['sha256']:
    paths.add(root/(Path(path).name if path.startswith('scripts/') else path))
paths.update(root/path for path in plan['required_admissions'])
paths.update(p for p in root.glob(name+'*.log') if p.is_file())
assert sum(p.name == 'output.png' and name in str(p) for p in paths) == plan['expected_outputs']
assert all(p.is_file() and not p.is_symlink() and 'cache' not in p.relative_to(root).parts for p in paths)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


members = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(paths)]
archive = root/(name+'-archive.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz', compresslevel=3) as stream:
    for path in sorted(paths):
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
result = dict(archive=archive.name, sha256=digest(archive), bytes=archive.stat().st_size,
              members=members, generated_images=plan['expected_outputs'], local_audit_pending=True)
manifest = root/(name+'-archive-manifest.json')
assert not manifest.exists()
manifest.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'members'}))
