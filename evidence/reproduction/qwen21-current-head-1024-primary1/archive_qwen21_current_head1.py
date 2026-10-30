"""Archive a complete prospective Qwen run before local numerical admission."""
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
assert re.fullmatch(r'qwen21-current-head-(512|1024)-(aa|primary)1', args.run)
plan_path = root/(args.run+'.json')
plan = json.loads(plan_path.read_text())
status_path = root/(args.run+'-status.json')
status = json.loads(status_path.read_text())
assert status['status'] == 'raw_complete_audit_pending'
assert len(status['completed']) == len(plan['order'])
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
paths = {plan_path, status_path, Path(__file__).resolve()}
for area, patterns in [
    ('block-runs', [args.run+'-*']),
    ('outputs/benchmarks', [args.run+'-*']),
    ('outputs/qualification', ['sgl-qwen21-base-*-'+args.run+'-*', 'omni-qwen21-base-*-'+args.run+'-*']),
]:
    for pattern in patterns:
        for item in (root/area).glob(pattern):
            paths.update([item] if item.is_file() else (p for p in item.rglob('*') if p.is_file()))
for name in plan['sha256']:
    paths.add(root/(Path(name).name if name.startswith('scripts/') else name))
paths.update(root/name for name in plan['required_admissions'])
paths.update(p for p in root.glob(args.run+'*.log') if p.is_file())
paths.update(p for p in (root/'outputs/diagnostics/qwen21-current-head-tests1').rglob('*') if p.is_file())
paths.add(root/'qwen21-current-head-tests1-status.json')
paths.add(root/'evidence/downloads/qwen21-current-head-restore1-modelscope-download-manifest.json')
expected_images = sum(len(plan['configurations'][config].get('tasks',plan['tasks'])) for config in plan['order'])*(4+plan['warmups']+plan['repeats'])
assert sum(p.name == 'output.png' and p.relative_to(root).parts[0] == 'outputs'
           for p in paths) == expected_images
assert all(p.is_file() and not p.is_symlink() and 'cache' not in p.relative_to(root).parts
           for p in paths)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


members = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=digest(p))
           for p in sorted(paths)]
archive = root/(args.run+'-archive.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz', compresslevel=3) as stream:
    for path in sorted(paths):
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
result = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=digest(archive),
              generated_images=expected_images, members=members,
              performance_admission_pending=True,
              note='Complete raw run before local audit; original references remain in their already verified archives. No failed performance gate is discarded, and no checkpoint weights are included.')
out = root/(args.run+'-archive-manifest.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: value for key, value in result.items() if key != 'members'}))
