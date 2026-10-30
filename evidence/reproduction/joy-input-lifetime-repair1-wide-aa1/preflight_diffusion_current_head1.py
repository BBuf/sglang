"""Check the explicitly selected native checkout and idle physical GPU pool."""
import hashlib
import json
import sys
from pathlib import Path

import diffusion_skill_env as env
import sglang

expected = Path(sys.argv[1])
root = env.get_repo_root()
assert root == expected
env.check_write_access(root)
assert env.pick_idle_gpus(2) == [0, 1]
print(json.dumps(dict(repo=str(root), module=sglang.__file__, writable=True,
                      idle_gpus=[0, 1], inventory=env.get_gpu_inventory(),
                      helper_sha256=hashlib.sha256(Path(env.__file__).read_bytes()).hexdigest())))
