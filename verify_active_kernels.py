"""Compare retained kernel bodies against the benchmark baseline (supplementary evidence)."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = "5916999afbc595de62dffca077fd8249634b4e74"
result = {}

def source(path):
    before = subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=root, text=True)
    return before, (root / path).read_text()

def compare(name, before, after):
    assert before == after, name
    result[name] = hashlib.sha256(after.encode()).hexdigest()

def cpp_body(text, name):
    match = re.search(r"\b" + name + r"\s*\(", text)
    start = text.index("{", match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[match.start():end]

py = "python/sglang/kernels/kda_kernels/residual_gate_add_jit.py"
before, after = source(py)
for name in ("_round16_f32", "_rga_transposed"):
    nodes = [next(n for n in ast.parse(t).body if isinstance(n, ast.FunctionDef) and n.name == name) for t in (before, after)]
    compare(name, *(ast.dump(n, include_attributes=False) for n in nodes))

checks = {
    "python/sglang/kernels/jit/csrc/diffusion/helios_qk_rope.cuh": ["helios_qk_rope_kernel"],
    "python/sglang/kernels/kda_kernels/csrc/diffusion/residual_gate_add.cuh": [
        "residual_gate_value", "residual_gate_add_vec_kernel", "residual_gate_add_broadcast_kernel", "residual_gate_add_scalar_kernel", "run"
    ],
}
for path, names in checks.items():
    before, after = source(path)
    for name in names:
        compare(f"{path}:{name}", cpp_body(before, name), cpp_body(after, name))
print(json.dumps(result, indent=2))
