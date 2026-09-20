"""Check whether the existing out-of-place QK/RoPE op can remove Joy's input copies."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys
import time

root = Path('/campaign')
done = root / 'validate-joy-qkv-cat-final.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
lock = (root / 'gpu.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0, str(root / 'baseline-super-current/python'))
import torch
from sglang.kernels.ops.diffusion import fused_qknorm_rope_out_of_place
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm_with_optional_rope

def measure(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(40):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000)
    return dict(median_us=statistics.median(values), samples_us=values)

records = []
with torch.inference_mode():
    for batch, tokens, heads in [(1,8048,32), (2,257,32), (1,1004,32)]:
        for dtype in [torch.bfloat16, torch.float16]:
            torch.manual_seed(42)
            packed = torch.randn(batch,tokens,3,heads,128,device='cuda',dtype=dtype)
            q,k,v = packed.unbind(2)
            norm_q = RMSNorm(128,eps=1e-6).to(device='cuda',dtype=dtype)
            norm_k = RMSNorm(128,eps=1e-6).to(device='cuda',dtype=dtype)
            norm_q.weight.copy_(torch.rand_like(norm_q.weight)+.5)
            norm_k.weight.copy_(torch.rand_like(norm_k.weight)+.5)
            theta = torch.randn(tokens,64,device='cuda')
            cache = torch.cat((theta.cos(),theta.sin()),-1).contiguous()
            def native():
                return apply_qk_norm_with_optional_rope(q.contiguous(),k.contiguous(),norm_q,norm_k,128,
                    cos_sin_cache=cache,is_neox=False,allow_inplace=True)
            def candidate():
                qo = torch.empty(q.shape,device=q.device,dtype=q.dtype)
                ko = torch.empty(k.shape,device=k.device,dtype=k.dtype)
                positions = torch.arange(tokens,device=q.device,dtype=torch.int64)
                if batch != 1:
                    positions = positions.repeat(batch)
                fused_qknorm_rope_out_of_place(q.view(-1,heads,128),k.view(-1,heads,128),
                    qo.view(-1,heads,128),ko.view(-1,heads,128),norm_q.weight,norm_k.weight,cache,positions,
                    is_neox=False,eps=1e-6,head_dim=128,rope_dim=128)
                return qo,ko
            for scale in [.001,1.,30.]:
                packed.copy_(torch.randn_like(packed)*scale)
                before=packed.clone()
                expected,actual=native(),candidate()
                assert torch.equal(before.view(torch.int16),packed.view(torch.int16))
                assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,expected))
            row=dict(batch=batch,tokens=tokens,heads=heads,dtype=str(dtype),exact=True,
                native=measure(native),candidate=measure(candidate))
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['native','candidate']}|{
                'native_us':row['native']['median_us'],'candidate_us':row['candidate']['median_us']}),flush=True)
(root / 'artifacts/joy-image-edit/outplace-qk-screen.json').write_text(json.dumps(records,indent=2))
