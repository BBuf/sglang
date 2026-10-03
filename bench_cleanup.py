"""Same-source, same-input baseline/candidate benchmark for the cleanup."""
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    residual_gate_add, modulate_scale_shift, usp_merge_heads,
    fused_causal_conv3d_cat_pad_cuda,
)
from sglang.multimodal_gen.runtime.models.dits.helios import HeliosSelfAttention


def rand(shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype)


def workloads():
    for label, shape, gate_shape, transposed in [
        ("flux_text", (1, 512, 3072), (1, 1, 3072), False),
        ("flux_image", (1, 4096, 3072), (1, 1, 3072), False),
        ("flux2_joint", (1, 4608, 6144), (1, 1, 6144), False),
        ("full_gate", (1, 512, 4096), (1, 512, 4096), False),
        ("per_token", (1, 2560, 512), (1, 2560, 1), False),
        ("odd_width", (2, 33, 65), (1, 1, 65), False),
        ("sana_transposed", (1, 32760, 2240), (1, 1, 2240), True),
        ("transposed_small", (1, 17, 65), (1, 1, 65), True),
        ("transposed_medium", (2, 513, 256), (1, 1, 256), True),
    ]:
        for dtype in (torch.bfloat16, torch.float16):
            r = rand((shape[0], shape[2], shape[1]), dtype).transpose(1, 2) if transposed else rand(shape, dtype)
            yield f"residual/{label}/{dtype}", residual_gate_add, (r, rand(shape, dtype), rand(gate_shape, dtype)), None
    for shape in ((1, 512, 3072), (1, 4096, 3072), (2, 1024, 3072)):
        yield f"modulate/{shape}", modulate_scale_shift, (rand(shape), rand((shape[0], shape[-1])), rand((shape[0], shape[-1]))), None
    for shape in ((4, 7936, 1, 14, 128), (2, 64, 3, 4, 64), (4, 33, 2, 4, 100)):
        yield f"usp/{shape}", usp_merge_heads, (rand(shape),), None
    attention = HeliosSelfAttention.__new__(HeliosSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.tp_rmsnorm = False
    for shape in ((1, 2160, 40, 128), (1, 8640, 40, 128), (2, 17, 8, 128)):
        freqs = rand((shape[0], shape[1], 256), torch.float32) * 0.01
        yield f"helios/{shape}", attention._apply_rotary_qk, (rand(shape), rand(shape), freqs), None
    for c, h, w in ((8, 6, 6), (512, 30, 52), (1024, 30, 52)):
        yield f"cat_pad/{c}/{h}/{w}", fused_causal_conv3d_cat_pad_cuda, (rand((1,c,1,h,w)), rand((1,c,1,h,w)), (1,1,1,1,1,0)), None


def fingerprints(out):
    if isinstance(out, torch.Tensor):
        raw=out.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(out.shape), stride=list(out.stride()), sha256=hashlib.sha256(raw).hexdigest())
    return [fingerprints(t) for t in out]


@torch.inference_mode()
def main():
    torch.manual_seed(20261003)
    result = dict(torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(0), arm=os.environ['ARM'], cases=[])
    for name, fn, args, kwargs in workloads():
        out = fn(*args)
        fingerprint = fingerprints(out)
        del out
        graph = marker.do_bench(fn, input_args=args, warmup_iters=10, replay_iters=100,
                               memory_output=None, disable_log_bandwidth=True)
        host = []
        for _ in range(3):
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(100): fn(*args)
            torch.cuda.synchronize()
            host.append((time.perf_counter()-start)*1e4)
        row=dict(name=name, graph_us=graph.times[0]*1e6, eager_us=statistics.median(host), output=fingerprint)
        result['cases'].append(row)
        print(json.dumps(row), flush=True)
    Path(os.environ['OUTPUT']).write_text(json.dumps(result, indent=2)+'\n')

if __name__=='__main__':
    main()
