"""One-off validation; intentionally kept out of the cleanup PR."""
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

BASE = "35f3c96ff4794a4de15daf12caad371084a037ee"
HEAD = "35bc03557c454add71b2805e8c221e68362336b1"
ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
os.chdir(ROOT)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == HEAD
print("VALIDATION_COMMITS", BASE, HEAD, flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import triton

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] >= 9
print("ENV", torch.__version__, triton.__version__, torch.cuda.get_device_properties(0), flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem", "--format=csv"], check=True)

tests = [
    "test/registered/kernels/ops/attention/test_gdn_decode_fused_proj_conv.py",
    "test/registered/kernels/ops/attention/test_vision_rope.py",
    "test/registered/kernels/ops/attention/test_mla_scatter_concat.py",
    "test/registered/unit/models/test_kimi_k3_vision.py",
    "test/registered/attention/test_gdn_noncontiguous_stride.py",
    "test/registered/dcp/test_trtllm_mla_family_dcp_metadata.py",
]
for test in tests:
    print("EXISTING_TEST", test, flush=True)
    subprocess.run([sys.executable, "-m", "pytest", test, "-q", "-rA"], check=True)

from sglang.kernels.ops.attention import triton_gdn_fused_proj as new_gdn
from sglang.kernels.ops.attention import vision_rope as new_rope
from sglang.kernels.ops.attention import set_mla_kv_concat_q as new_mla
from sglang.srt.models import kimi_k3_vl as new_kimi

subprocess.run(["git", "fetch", "--depth=1", "https://github.com/sgl-project/sglang.git", BASE], check=True)
tmp = Path(tempfile.mkdtemp(prefix="llm_vlm_cleanup_ab_"))

def baseline(name, path):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"])
    dest = tmp / f"{name}.py"
    dest.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, dest)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

old_gdn = baseline("baseline_gdn", "python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py")
old_rope = baseline("baseline_rope", "python/sglang/kernels/ops/attention/vision_rope.py")
old_mla = baseline("baseline_mla", "python/sglang/kernels/ops/attention/set_mla_kv_concat_q.py")
sys.modules["sglang.kernels.ops.attention.vision_rope"] = old_rope
try:
    old_kimi = baseline("baseline_kimi", "python/sglang/srt/models/kimi_k3_vl.py")
finally:
    sys.modules["sglang.kernels.ops.attention.vision_rope"] = new_rope

for mod in (old_rope, new_rope):
    assert mod.precompile_fused_qk_complex_rope(num_heads=12, head_dim=128, dtype=torch.bfloat16, device=torch.device("cuda"))
    assert not mod.precompile_fused_qk_complex_rope(num_heads=12, head_dim=128, dtype=torch.bfloat16, device=torch.device("cpu"))
print("ROPE_PRECOMPILE_PARITY", "implicit CUDA device and CPU fallback", flush=True)

torch.manual_seed(20261004)
random.seed(20261004)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
rows = []
dispatch_rows = []
# 256 MiB > the entire L2 capacity of the H100 used for this run. The eviction
# pass is replayed before EVERY individual operation; it is outside the events.
# It flushes the device L2, not allocator/JIT/CPU caches, and does not flush
# between the constituent kernels of an encoder forward.
eviction = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
print("L2_POLICY", json.dumps({"bytes": eviction.numel(), "scope": "before every individual graph replay operation; before an entire encoder forward, not between its kernels", "outside_event_interval": True, "samples_per_side": 150}), flush=True)

def exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
        return
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (a.shape, a.dtype)

def cold_graph(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        eviction.zero_()
        start.record()
        out = fn()
        end.record()
    return graph, start, end, out

def compare(name, funcs, selectors=None):
    exact(funcs[0](), funcs[1]())
    graphs = [cold_graph(f) for f in funcs]
    samples = [[], []]
    for rep in range(160):
        order = [0, 1]
        random.shuffle(order)
        for i in order:
            graph, start, end, _ = graphs[i]
            graph.replay()
            end.synchronize()
            if rep >= 10:
                samples[i].append(start.elapsed_time(end) * 1000)
    old_us, new_us = map(statistics.median, samples)
    row = {"case": name, "base_us": old_us, "candidate_us": new_us, "ratio": new_us / old_us,
           "base_p10_p90": [sorted(samples[0])[15], sorted(samples[0])[134]],
           "candidate_p10_p90": [sorted(samples[1])[15], sorted(samples[1])[134]]}
    rows.append(row)
    print("COLD_L2_RESULT", json.dumps(row), flush=True)
    if selectors:
        timing = [[], []]
        assert selectors[0]() == selectors[1]()
        for rep in range(40):
            for i in ([0, 1] if rep % 2 else [1, 0]):
                t = time.perf_counter_ns()
                for _ in range(100):
                    selectors[i]()
                timing[i].append((time.perf_counter_ns() - t) / 100 / 1000)
        row = {"case": name, "base_us": statistics.median(timing[0]), "candidate_us": statistics.median(timing[1])}
        dispatch_rows.append(row)
        print("HOST_SELECTOR_RESULT", json.dumps(row), flush=True)

for batch, width, dtype in itertools.product((1, 17, 64), (2, 3, 4), (torch.float16, torch.bfloat16)):
    qkv_dim, v_dim, heads, head_dim = 1280, 1024, 8, 128
    kwargs = dict(qkv_dim=qkv_dim, v_dim=v_dim, num_v_heads=heads, head_v_dim=head_dim, activation="silu")
    qkvz = torch.randn(batch, qkv_dim + v_dim, device="cuda", dtype=dtype)
    ba = torch.randn(batch, 2 * heads, device="cuda", dtype=torch.float32 if width == 3 else dtype)
    weight = torch.randn(qkv_dim, width, device="cuda", dtype=dtype) * 0.1
    bias = torch.randn(qkv_dim, device="cuda", dtype=dtype) * 0.1 if width == 3 else None
    state = torch.randn(batch + 3, 5, qkv_dim * 2, device="cuda", dtype=dtype)[:, :, ::2].transpose(1, 2)
    states = [torch.empty_strided(state.shape, state.stride(), dtype=dtype, device="cuda").copy_(state) for _ in range(2)]
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    if batch > 1:
        indices[-1] = -1
    def gdn_call(mod, cache):
        outputs = mod.fused_qkvzba_causal_conv1d_update_contiguous(qkvz, ba, cache, weight, bias, indices, **kwargs)
        return (*outputs, cache)
    funcs = [lambda: gdn_call(old_gdn, states[0]), lambda: gdn_call(new_gdn, states[1])]
    selectors = [
        lambda: old_gdn.can_use_fused_qkvzba_causal_conv1d_update_contiguous(qkvz, ba, states[0], weight, bias, indices, qkv_dim=qkv_dim, v_dim=v_dim, num_v_heads=heads, activation="silu")[0],
        lambda: new_gdn.can_use_fused_qkvzba_causal_conv1d_update_contiguous(qkvz, ba, states[1], weight, bias, activation="silu")[0],
    ]
    assert all(f() for f in selectors)
    compare(f"gdn/B{batch}/W{width}/{dtype}/strided_state", funcs, selectors)

for tokens, dtype, strided in itertools.product((1, 480, 2048), (torch.float16, torch.bfloat16), (False, True)):
    qkv = torch.randn(tokens, 3, 12, 128, dtype=dtype, device="cuda")
    q, k, _ = qkv.unbind(1)
    if not strided:
        q, k = q.contiguous(), k.contiguous()
    angles = torch.randn(tokens, 64, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    compare(f"rope/T{tokens}/{dtype}/strided{strided}", [lambda: old_rope.apply_fused_qk_complex_rope(q, k, freqs), lambda: new_rope.apply_fused_qk_complex_rope(q, k, freqs)], [lambda: old_kimi._can_use_fused_rope(q, freqs), lambda: new_rope.can_use_fused_qk_complex_rope(dtype=q.dtype, device=q.device, freqs_cis=freqs)])

for batch, fp8 in itertools.product((1, 64, 256), (False, True)):
    latent = torch.randn(batch, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    query = torch.randn(batch, 8, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    kn, kr, qn, qr = latent[:, :512], latent[:, 512:], query[..., :512], query[..., 512:]
    loc = torch.randperm(1024, device="cuda")[:batch]
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    pools = [torch.zeros(1024, 576, device="cuda", dtype=dtype) for _ in range(2)]
    def mla_call(mod, pool):
        fn = mod.set_mla_kv_concat_q_fp8 if fp8 else mod.set_mla_kv_concat_q
        out = fn(pool, loc, kn, kr, qn, qr)
        return out, pool
    selectors = [lambda: (old_mla.covered_fp8 if fp8 else old_mla.covered)(pools[0], loc, kn, kr, qn, qr), lambda: (new_mla.covered_fp8 if fp8 else new_mla.covered)(pools[1], loc, kn, kr, qn, qr)]
    assert all(f() for f in selectors)
    compare(f"mla/B{batch}/fp8{fp8}", [lambda: mla_call(old_mla, pools[0]), lambda: mla_call(new_mla, pools[1])], selectors)
    # Valid layouts which must still use the unfused fallback.
    bad_loc = torch.arange(batch * 2, device="cuda")[::2]
    unaligned = torch.zeros(1024, 577, device="cuda", dtype=dtype)[:, 1:]
    for pool, positions, key in ((unaligned, loc, kn), (pools[0], bad_loc, kn), (pools[0], loc, kn.float())):
        before = (old_mla.covered_fp8 if fp8 else old_mla.covered)(pool, positions, key, kr, qn, qr)
        after = (new_mla.covered_fp8 if fp8 else new_mla.covered)(pool, positions, key, kr, qn, qr)
        assert before == after

# Exercise the actual encoder, including QKV projection, shared metadata, RoPE,
# SDPA, MLP and final normalization, with identical random weights and input.
# This is integration parity, not checkpoint task accuracy or serving throughput.
config = dict(num_heads=2, hidden_dim=128, qkv_hidden_size=256, mlp_dim=256, norm_type="rmsnorm", activation=torch.nn.functional.gelu, attn_bias=False, linear_bias=False)
encoders = [m.MoonViT3dEncoder(hidden_dim=128, num_layers=2, block_cfg=config).cuda().bfloat16().eval() for m in (old_kimi, new_kimi)]
encoders[1].load_state_dict(encoders[0].state_dict())
for shapes in (((1, 4, 4),), ((1, 4, 4), (2, 2, 4))):
    grid = torch.tensor(shapes, device="cuda")
    tokens = sum(t * h * w for t, h, w in shapes)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    metadata = [m.prepare_forward_metadata(grid_thws=grid, total_tokens=tokens, dtype=x.dtype, device=x.device, grid_thw_list=shapes) for m in encoders]
    assert all(item.use_fused_rope for item in metadata)
    outputs = [m(x, grid, grid_thw_list=shapes) for m in encoders]
    exact(*outputs)
    for i, m in enumerate(encoders):
        exact(outputs[i], m(x, grid, forward_metadata=metadata[i]))
    compare(f"kimi_encoder/{shapes}", [lambda: encoders[0](x, grid, forward_metadata=metadata[0]), lambda: encoders[1](x, grid, forward_metadata=metadata[1])])
    print("KIMI_ENCODER_PARITY", shapes, "bitwise, fused_rope=True, prepared and unprepared", flush=True)

result = dict(base=BASE, head=HEAD, gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__, l2_flush_bytes=eviction.numel(), rows=rows, dispatch_rows=dispatch_rows)
(tmp / "results.json").write_text(json.dumps(result, indent=2))
print("VALIDATION_JSON", json.dumps(result), flush=True)
print("VALIDATION_COMPLETE", len(rows), "cold-L2 A/B cases with bitwise output/state parity", flush=True)
