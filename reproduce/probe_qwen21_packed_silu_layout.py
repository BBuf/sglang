"""Offline launch-layout experiment for the observed packed-SiLU profile cost."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from sglang.kernels.numerics import round_bf16_to_fp32
from sglang.kernels.ops.diffusion.activation.silu_mul_bitexact import _packed_silu_mul_kernel


@triton.jit
def row_major(out, x, rows, stride, out_stride, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (cols < D)
    a = tl.load(x + row * stride + cols, mask, other=0).to(tl.float32)
    b = tl.load(x + row * stride + D + cols, mask, other=0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out + row * out_stride + cols, s * b, mask)


def measure(fn, repeats=30):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeats):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / repeats


torch.set_grad_enabled(False)
torch.manual_seed(1101)
records = []
for rows in [4096, 8192]:
    x = torch.randn(1, rows, 24576, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(1, rows, 12288, dtype=x.dtype, device=x.device)
    reference = torch.nn.functional.silu(x[..., :12288]) * x[..., 12288:]
    funcs = {}
    for order in ['column', 'row']:
        for block in [256, 512, 1024, 2048, 4096, 8192]:
            for warps in [4, 8]:
                kernel = _packed_silu_mul_kernel if order == 'column' else row_major
                grid = (rows, triton.cdiv(12288, block)) if order == 'column' else (triton.cdiv(12288, block), rows)
                def fn(kernel=kernel, grid=grid, block=block, warps=warps):
                    return kernel[grid](out, x, rows, 24576, 12288, D=12288, BLOCK=block, num_warps=warps)
                label = f'{order}-b{block}-w{warps}'
                fn()
                assert torch.equal(out, reference), label
                funcs[label] = fn
    baseline = funcs['column-b1024-w4']
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        baseline()
        torch.cuda.synchronize()
    samples = {label: [] for label in funcs}
    for labels in [list(funcs), list(reversed(funcs)), list(funcs)]:
        for label in labels:
            for _ in range(5):
                funcs[label]()
            samples[label].append(measure(funcs[label]))
    values = [dict(label=label, ms=v, median_ms=statistics.median(v), exact=True) for label, v in samples.items()]
    values.sort(key=lambda v:v['median_ms'])
    records.append(dict(rows=rows, variants=values))
    print(json.dumps(dict(rows=rows, best=values[:6], baseline=next(v for v in values if v['label']=='column-b1024-w4'))), flush=True)
dest = Path('/task/evidence/preflight/qwen21-packed-silu-layout-probe1.json')
assert not dest.exists()
dest.write_text(json.dumps(dict(diagnostic_only=True, all_variants_exact=True, torch=torch.__version__, gpu=torch.cuda.get_device_name(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records), indent=2)+'\n')
