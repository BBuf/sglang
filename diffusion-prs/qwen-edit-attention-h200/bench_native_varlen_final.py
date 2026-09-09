"""Benchmark the production packed-SDPA path, including the actual Qwen window lengths."""
import json
import subprocess
import sglang
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl


def single(q, k, v, scale, causal=False):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
        v.transpose(1, 2), dropout_p=0.0, is_causal=causal, scale=scale).transpose(1, 2)


def baseline(q, k, v, bounds, scale, causal=False):
    output = torch.empty_like(q)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start != stop:
            output[start:stop].copy_(single(q[start:stop][None], k[start:stop][None],
                v[start:stop][None], scale, causal)[0])
    return output


def bench(fn, iterations=30):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations): fn()
    stop.record()
    stop.synchronize()
    return dict(cuda_ms=start.elapsed_time(stop)/iterations,
                wall_ms=1000*(time.perf_counter()-wall)/iterations)


source_repo = Path(sglang.__file__).resolve().parents[2]
source_commit = subprocess.check_output(['git', '-C', str(source_repo), 'rev-parse', 'HEAD'], text=True).strip()
assert source_commit == '60def09d35c8912db76144e7ca6691dca554bc94'
assert not subprocess.check_output(['git', '-C', str(source_repo), 'status', '--porcelain'], text=True).strip()
torch.manual_seed(42)
torch.backends.cuda.enable_cudnn_sdp(False)
rows = []
for dtype in [torch.bfloat16, torch.float16]:
    for lengths, heads, dim, strided, causal in [
        (([64]*7+[32])*11+[16]*7+[8],16,80,True,False), ([64]*96,16,80,True,False), ([64]*12+[32]+[64]*12+[32],16,80,True,False),
        ([64]*7+[0,32,32,0,64],16,80,False,False), ([6144],16,80,True,False),
        ([32]*8,8,64,True,True), ([128]*4,16,128,False,False)]:
        bounds=[0]
        for length in lengths:bounds.append(bounds[-1]+length)
        storage=torch.randn(bounds[-1],3,heads,dim,device='cuda',dtype=dtype)
        q,k,v=storage.unbind(1)
        q,k=q.contiguous(),k.contiguous()
        if not strided:q,k,v=[x.contiguous() for x in (q,k,v)]
        scale=dim**-0.5
        impl=SDPAImpl(heads,dim,causal=causal,softmax_scale=scale)
        cu=torch.tensor(bounds,device="cuda",dtype=torch.int32)
        def run_candidate():
            return impl.forward_varlen(q,k,v,cu_seqlens=cu,max_seqlen=max(lengths),cu_seqlens_host=tuple(bounds))
        def run_reference():
            return baseline(q,k,v,bounds,scale,causal)
        with torch.no_grad():
            a=baseline(q,k,v,bounds,scale,causal)
            b=run_candidate()
            exact=torch.equal(a,b)
            max_abs=(a.float()-b.float()).abs().max().item()
            times={}
            if exact:
                for tag, fn in [('a1',run_reference),('b1',run_candidate),('b2',run_candidate),('a2',run_reference)]:
                    times[tag]=bench(fn)
            profiles={}
            if dtype == torch.bfloat16 and len(lengths)==96 and sum(lengths)==5400:
                for tag,fn in [('reference',run_reference),('candidate',run_candidate)]:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        fn();torch.cuda.synchronize()
                    path=Path('/data/goals/sglang-h200-diffusion-20260907/results')/f'qwen-edit-native-varlen-final-micro-{tag}.trace.json'
                    prof.export_chrome_trace(str(path))
                    events=json.loads(path.read_text())['traceEvents']
                    kernels=[e for e in events if e.get('cat')=='kernel']
                    profiles[tag]=dict(kernel_count=len(kernels),cumulative_kernel_ms=sum(e['dur'] for e in kernels)/1000,trace=str(path))
        row=dict(dtype=str(dtype),lengths=lengths,heads=heads,head_dim=dim,strided=strided,
                 causal=causal,exact=exact,max_abs=max_abs,times=times,profiles=profiles)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del storage,q,k,v,a,b
root=Path('/data/goals/sglang-h200-diffusion-20260907')
(root/'results/qwen-edit-native-varlen-final-production-microbenchmark.json').write_text(json.dumps(dict(
    scope="Eager CUDA-event timings include host launch gaps; direct kernel time comes from the separate tiny traces.",source_commit=source_commit,torch=torch.__version__,device=torch.cuda.get_device_name(),cudnn_enabled=False,rows=rows),indent=2)+'\n')

assert all(row["exact"] for row in rows)
