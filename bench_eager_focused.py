import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from sglang.kernels.ops.diffusion import residual_gate_add

@torch.inference_mode()
def main():
    torch.manual_seed(20261003)
    r = torch.randn(1, 512, 3072, device="cuda", dtype=torch.bfloat16)
    u = torch.randn_like(r)
    g = torch.randn(1, 1, 3072, device="cuda", dtype=r.dtype)
    gc.disable()
    for _ in range(1000):
        residual_gate_add(r, u, g)
    values = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            residual_gate_add(r, u, g)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    result = dict(arm=os.environ["ARM"], name="residual/flux_text/torch.bfloat16", eager_us=statistics.median(values), samples_us=values)
    print(json.dumps(result), flush=True)
    Path(os.environ["OUTPUT"]).write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
