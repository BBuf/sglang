"""ABBA microbenchmark of the exact committed FastH3 wrapper functions."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

root = Path('/campaign')
lock = (root/'gpu.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
candidate, baseline = root/'candidate-fasth3-swiglu', root/'baseline'
sys.path.insert(0, str(candidate/'python'))
import torch
import triton
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _silu_mul as after

name = 'campaign_baseline_h3'
spec = importlib.util.spec_from_file_location(name, baseline/'python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py')
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
before = module._silu_mul
torch.manual_seed(2026)
rows = []
with torch.no_grad():
    for tokens in (1, 1024, 36704):
        hidden = torch.randn(tokens, 28672, device='cuda', dtype=torch.bfloat16)
        saved = hidden.clone()
        a, b = before(hidden, reuse_input=False), after(hidden, reuse_input=False)
        assert torch.equal(a, b) and torch.equal(hidden, saved)
        assert b.is_contiguous() and b.data_ptr() != hidden.data_ptr()
        del a, b, saved
        times = {}
        for label, fn in [('a1', before), ('b1', after), ('b2', after), ('a2', before)]:
            times[label] = triton.testing.do_bench(lambda: fn(hidden, reuse_input=False), warmup=100, rep=300)
        row = dict(shape=list(hidden.shape), exact=True, input_preserved=True, mean_ms=times,
                   baseline_mean_ms=(times['a1']+times['a2'])/2,
                   candidate_mean_ms=(times['b1']+times['b2'])/2)
        rows.append(row)
        print(json.dumps(row), flush=True)
result = dict(torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(),
              baseline=subprocess.check_output(['git', '-C', str(baseline), 'rev-parse', 'HEAD'], text=True).strip(),
              candidate=subprocess.check_output(['git', '-C', str(candidate), 'rev-parse', 'HEAD'], text=True).strip(),
              timing='triton.testing.do_bench mean milliseconds, warmup100ms rep300ms per ABBA cell', rows=rows)
(root/'artifacts/fasth3/swiglu-committed-microbench.json').write_text(json.dumps(result, indent=2))
