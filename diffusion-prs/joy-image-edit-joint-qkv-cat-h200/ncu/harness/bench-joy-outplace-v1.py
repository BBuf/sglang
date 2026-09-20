"""Measure Joy image QK input copies/norm/RoPE plus all three joint cats."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('--ncu', choices=['baseline','candidate'])
args = p.parse_args()
root = Path('/campaign')
repo = root / 'candidate-joy-outplace-v1'
os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(repo/'python'), FLASHINFER_DISABLE_VERSION_CHECK='1')
sys.path.insert(0, str(repo/'python'))
import torch
from sglang.multimodal_gen.runtime.models.dits.joy_image import _joy_image_qk_rope, _joy_joint_qkv
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm_with_optional_rope

def measure(fn):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(40):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000)
    return dict(median_us=statistics.median(samples),samples_us=samples)

with (root/'gpu.lock').open('a') as lock, torch.inference_mode():
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.manual_seed(42)
    records=[]
    shapes=[(1,8048,1004,32)] if args.ncu else [(1,8048,1004,32),(1,4096,512,32),(1,2048,256,32),(2,257,13,32)]
    for batch, image_tokens, text_tokens, heads in shapes:
        image=torch.randn(batch,image_tokens,3,heads,128,device='cuda',dtype=torch.bfloat16)
        text=torch.randn(batch,text_tokens,3,heads,128,device='cuda',dtype=torch.bfloat16)
        iq,ik,iv=image.unbind(2)
        tq,tk,tv=text.unbind(2)
        tq,tk=tq.contiguous(),tk.contiguous()
        qn=RMSNorm(128,eps=1e-6).to(device='cuda',dtype=iq.dtype)
        kn=RMSNorm(128,eps=1e-6).to(device='cuda',dtype=iq.dtype)
        qn.weight.copy_(torch.rand_like(qn.weight)+.5)
        kn.weight.copy_(torch.rand_like(kn.weight)+.5)
        angles=torch.randn(image_tokens,64,device='cuda')
        cache=torch.cat((angles.cos(),angles.sin()),dim=-1)
        def baseline():
            q,k=apply_qk_norm_with_optional_rope(iq.contiguous(),ik.contiguous(),qn,kn,128,
                cos_sin_cache=cache,is_neox=False)
            return torch.cat((q,tq),1),torch.cat((k,tk),1),torch.cat((iv,tv),1)
        def candidate():
            q,k=_joy_image_qk_rope(iq,ik,qn,kn,cache,None)
            return _joy_joint_qkv(q,k,iv,tq,tk,tv)
        before=image.clone()
        expected,actual=baseline(),candidate()
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(expected,actual))
        assert torch.equal(before.view(torch.int16),image.view(torch.int16))
        if args.ncu:
            fn=baseline if args.ncu=='baseline' else candidate
            for _ in range(10):fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push('profile');fn();torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        else:
            row=dict(batch=batch,image_tokens=image_tokens,text_tokens=text_tokens,heads=heads,dim=128,eager={},standalone_graph={})
            for arm,fn in [('baseline',baseline),('candidate',candidate)]:
                row['eager'][arm]=measure(fn)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=fn()
                row['standalone_graph'][arm]=measure(graph.replay)
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['eager','standalone_graph']}|{
                mode:{arm:values['median_us'] for arm,values in row[mode].items()} for mode in ['eager','standalone_graph']}),flush=True)
    if not args.ncu:
        (root/'artifacts/joy-image-edit/outplace-chain-microbench-v1.json').write_text(json.dumps(records,indent=2))
