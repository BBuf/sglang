"""Verify every archive member before restoring local evidence; never overwrite differences."""
import argparse
import datetime
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('name')
args = parser.parse_args()
folder = ROOT / 'evidence/checkpoints'
receipt = json.loads((folder / (args.name + '-archive-manifest.json')).read_text())
archive = folder / receipt['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
expected = {x['path']: x for x in receipt['members']}
seen = set()
with tarfile.open(archive) as stream:
    for member in stream:
        assert member.isfile() and member.name in expected
        data = stream.extractfile(member).read()
        row = expected[member.name]
        assert len(data) == row['bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        relative = Path(member.name)
        assert '..' not in relative.parts and not relative.is_absolute()
        if relative.parts[0] == 'outputs':
            dest = ROOT / 'evidence' / Path(*relative.parts[1:])
        elif relative.parts[0] == 'block-runs':
            dest = ROOT / 'evidence' / relative
        elif relative.parts[0] == 'evidence':
            dest = ROOT / relative
        else:
            raise AssertionError(member.name)
        if dest.exists():
            assert dest.read_bytes() == data, str(dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        seen.add(member.name)
assert seen == set(expected)
result = dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              archive=archive.name, sha256=receipt['sha256'], members=len(seen),
              all_member_hashes_verified=True, all_members_extracted=True)
(folder / (args.name + '-local-verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
