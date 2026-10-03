import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
mismatches = [name for name, expected in manifest["files"].items()
              if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
result = dict(verified_files=len(manifest["files"]), mismatches=mismatches,
              candidate=manifest["candidate"])
print(json.dumps(result, indent=2))
sys.exit(bool(mismatches))
