"""Production facade benchmark on actual Qwen Edit shapes from its full trace.

Restoring reference input happens outside measured intervals. No model mutation.
"""
import argparse
import json
import sglang
import statistics
import time
from pathlib import Path

import torch

from sglang.kernels.ops.diffusion import try_fused_qwen_qkv_epilogue
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm_with_optional_rope

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='production')
args = parser.parse_args()
root = Path('/data/goals/sglang-h200-diffusion-20260907')
torch.manual_seed(42)
assert torch.cuda.get_device_capability() == (9, 0)
import subprocess
source_commit = subprocess.check_output(["git", "-C", str(Path(sglang.__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip()
rows = []
with torch.inference_mode():
    for ni, nt, heads, packed in [(17, 7, 4, False), (17, 7, 4, True),
                                  (8152, 1365, 24, False), (8152, 1358, 24, False),
                                  (8152, 1365, 24, True), (16384, 512, 24, False)]:
        def inputs(n):
            if packed:
                storage = torch.randn(1, n, 3 * heads * 128, device='cuda', dtype=torch.bfloat16)
                return [t.unflatten(-1, (heads, 128)) for t in storage.chunk(3, dim=-1)]
            return [torch.randn(1, n, heads, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        img, txt = inputs(ni), inputs(nt)
        norms = [RMSNorm(128, eps=1e-6).to(device='cuda', dtype=torch.bfloat16) for _ in range(4)]
        for norm in norms:
            norm.weight.copy_(torch.randn_like(norm.weight))
        def cache(n):
            angles = torch.randn(n, 64, device='cuda')
            return torch.cat([angles.cos(), angles.sin()], -1).contiguous()
        ic, tc = cache(ni), cache(nt)
        ri, rt = [[t.contiguous().clone() for t in tensors] for tensors in (img, txt)]
        def restore():
            for target, source in zip((*ri, *rt), (*img, *txt)):
                target.copy_(source)
        def reference():
            iq, ik = apply_qk_norm_with_optional_rope(ri[0], ri[1], norms[0], norms[1], 128, ic, is_neox=False)
            tq, tk = apply_qk_norm_with_optional_rope(rt[0], rt[1], norms[2], norms[3], 128, tc, is_neox=False)
            return torch.cat([tq, iq], 1), torch.cat([tk, ik], 1), torch.cat([rt[2], ri[2]], 1)
        def candidate():
            outputs = try_fused_qwen_qkv_epilogue(*img, *txt, *[n.weight for n in norms], ic, tc, 1e-6, 1e-6)
            assert outputs is not None
            return outputs
        restore()
        expected, actual = reference(), candidate()
        exact = [torch.equal(a, b) for a, b in zip(expected, actual)]
        max_abs = [(a.float()-b.float()).abs().max().item() for a, b in zip(expected, actual)]
        mismatches = [torch.count_nonzero(a != b).item() for a, b in zip(expected, actual)]
        times = {}
        if all(exact):
            for tag, fn in [('a1', reference), ('b1', candidate), ('b2', candidate), ('a2', reference)]:
                measurements, walls = [], []
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for iteration in range(35):
                    restore()
                    torch.cuda.synchronize()
                    start.record()
                    wall = time.perf_counter()
                    output = fn()
                    stop.record()
                    stop.synchronize()
                    if iteration >= 5:
                        measurements.append(start.elapsed_time(stop))
                        walls.append((time.perf_counter()-wall)*1000)
                times[tag] = dict(cuda_median_ms=statistics.median(measurements), wall_median_ms=statistics.median(walls))
        profiles = {}
        if ni == 8152 and nt == 1365 and not packed:
            for tag, fn in [('reference', reference), ('candidate', candidate)]:
                restore()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    output = fn()
                    torch.cuda.synchronize()
                trace_path = root/'results'/f'qwen-edit-qkv-{args.tag}-{tag}.trace.json'
                prof.export_chrome_trace(str(trace_path))
                data = json.loads(trace_path.read_text())
                kernels = [e for e in data['traceEvents'] if e.get('cat')=='kernel']
                profiles[tag] = dict(kernel_count=len(kernels), cumulative_kernel_ms=sum(e['dur'] for e in kernels)/1000,
                    kernels=[dict(name=e['name'], duration_us=e['dur']) for e in kernels])
        row = dict(profiles=profiles, img_tokens=ni, txt_tokens=nt, heads=heads, packed=packed, exact=exact, max_abs=max_abs, mismatches=mismatches, times=times)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root/f'results/qwen-edit-qkv-{args.tag}-microbenchmark.json').write_text(json.dumps(dict(
            source_commit=source_commit, note='Production facade on SM90. Actual workload shapes8152/1365 and8152/1358 from original profile. CUDA event timings include host launch gaps. Reference copies to restore its in-place inputs are excluded from timing. Packed reference contiguous-copy costs are also excluded (conservative).',
            device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__, rows=rows), indent=2)+'\n')
        del img, txt, ri, rt, expected, actual
