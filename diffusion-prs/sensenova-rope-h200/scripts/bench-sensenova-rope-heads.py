"""ABBA microbench of the exact committed before/after SenseNova functions."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/campaign')
deadline=time.monotonic()+2400
while not (root/'sensenova-rope-abba.exit').exists():
    if time.monotonic()>deadline: raise TimeoutError('waiting for formal request runs')
    time.sleep(5)
assert (root/'sensenova-rope-abba.exit').read_text().strip()=='0'
lock=(root/'gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
candidate=root/'candidate-sensenova-rope'
baseline=root/'baseline-sensenova-profile'
sys.path.insert(0,str(candidate/'python'))
import torch
import triton
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import apply_rotary_pos_emb as after
name='sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.campaign_baseline_qwen3'
file=baseline/'python/sglang/multimodal_gen/runtime/models/sensenova_u1/neo_unify/modeling_qwen3.py'
spec=importlib.util.spec_from_file_location(name,file)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before=module.apply_rotary_pos_emb
torch.manual_seed(2026)
rows=[]
with torch.no_grad():
    for seq in (37,128,4096):
        for axis in ('t','h','w'):
            q=torch.randn(1,seq,32,64,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(1,seq,8,64,device='cuda',dtype=torch.bfloat16)
            if axis!='t':
                offset=0 if axis=='h' else 32
                q,k=q[...,offset:offset+32],k[...,offset:offset+32]
            q,k=q.transpose(1,2),k.transpose(1,2)
            angles=torch.randn(1,seq,q.shape[-1],device='cuda',dtype=torch.float32)
            cos,sin=angles.cos().to(torch.bfloat16),angles.sin().to(torch.bfloat16)
            a,b=before(q,k,cos,sin),after(q,k,cos,sin)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            times={}
            for label,fn in [('a1',before),('b1',after),('b2',after),('a2',before)]:
                times[label]=triton.testing.do_bench(lambda:fn(q,k,cos,sin),warmup=100,rep=300)
            row=dict(seq=seq,axis=axis,q_shape=list(q.shape),q_stride=list(q.stride()),
                     k_stride=list(k.stride()),byte_exact=True,mean_ms=times,
                     baseline_mean_ms=(times['a1']+times['a2'])/2,candidate_mean_ms=(times['b1']+times['b2'])/2)
            rows.append(row)
            print(json.dumps(row),flush=True)
out=dict(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),
    baseline=subprocess.check_output(['git','-C',str(baseline),'rev-parse','HEAD'],text=True).strip(),
    candidate=subprocess.check_output(['git','-C',str(candidate),'rev-parse','HEAD'],text=True).strip(),
    timing='triton.testing.do_bench mean milliseconds, warmup100ms rep300ms per cell, ABBA',rows=rows)
(root/'artifacts/sensenova-u1/rope-committed-microbench.json').write_text(json.dumps(out,indent=2))
