"""Real checkpoint MLP packing screen, motivated by native full-model profiles.

This is a diagnostic microbenchmark, never a model-level performance claim.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from sglang.kernels.ops.diffusion import (
    fused_packed_silu_mul_bitexact,
    fused_silu_mul_bitexact,
)

ROOT = Path('/task')
CACHE = ROOT/'cache/common/Qwen-Image-2.1/transformer'
torch.set_grad_enabled(False)
torch.set_num_threads(8)
index_path = CACHE/'diffusion_pytorch_model.safetensors.index.json'
index = json.loads(index_path.read_text())
weights = {}
for kind in ['gate_layer','proj','out']:
    name = f'transformer_blocks.0.img_mlp.{kind}.weight'
    with safe_open(CACHE/index['weight_map'][name], framework='pt', device='cpu') as source:
        weights[kind] = source.get_tensor(name).to(device='cuda',dtype=torch.bfloat16)
packed = torch.cat([weights['gate_layer'],weights['proj']],dim=0)
assert weights['gate_layer'].shape == (12288,4096)
assert weights['out'].shape == (4096,12288)


def original(x):
    gate = F.linear(x, weights['gate_layer'])
    up = F.linear(x, weights['proj'])
    return F.linear(fused_silu_mul_bitexact(gate,up), weights['out'])


def combined(x):
    return F.linear(fused_packed_silu_mul_bitexact(F.linear(x,packed)),weights['out'])


def measure(fn,x,repeats):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):fn(x)
    end.record();end.synchronize()
    return begin.elapsed_time(end)/repeats


rows=[]
for tokens in [1024,2304]:
    x=torch.randn(1,tokens,4096,generator=torch.Generator().manual_seed(1101),dtype=torch.bfloat16).cuda()
    a,b=original(x),combined(x)
    err=(a.float()-b.float()).abs()
    blocks=[]
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        original(x);torch.cuda.synchronize()
    for label,fn in [('aa0',original),('aa1',original)]+[('original',original),('combined',combined),('combined',combined),('original',original)]*3:
        for _ in range(10):fn(x)
        torch.cuda.synchronize()
        values=[measure(fn,x,20) for _ in range(9)]
        blocks.append(dict(arm=label,individual_mean_ms=values,median_ms=statistics.median(values)))
    ratios=[v for i in [2,6,10] for v in (blocks[i]['median_ms']/blocks[i+1]['median_ms'],blocks[i+3]['median_ms']/blocks[i+2]['median_ms'])]
    row=dict(tokens=tokens,shape=list(x.shape),pixel_claim=False,activation_source='seeded random BF16; real layer0 weights',
             tensor_exact=torch.equal(a,b),max_abs=float(err.max()),relative_l2=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(a.float())),
             aa_ratio=blocks[0]['median_ms']/blocks[1]['median_ms'],blocks=blocks,paired_ratios=ratios,geometric_mean_ratio=statistics.geometric_mean(ratios))
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='blocks'}),flush=True)
result=dict(diagnostic_only=True,not_model_performance=True,source='native Qwen Base layer0 checkpoint weights',
            checkpoint='d26bb61231c349cf6b7896fa83353113880e1ba3',index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,results=rows,
            interpretation='Only pursue a model candidate if the real denoise shape improves; retain losses and require full-model quality/AA/paired primary measurements.')
out=ROOT/'evidence/preflight/qwen21-bf16-gate-up-smaller-target-probe1.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
