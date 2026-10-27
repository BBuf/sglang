"""Actual Joy weights/shapes: diagnostic only, not model performance evidence."""
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

root = Path('/task')
model = root/'cache/common/JoyAI-Image-Edit-Diffusers/transformer'
index = json.loads((model/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
torch.set_grad_enabled(False)
torch.manual_seed(1101)
rows = []
for branch, tokens in [('img', 8192), ('txt', 1053), ('txt', 1033)]:
    prefix = f'double_blocks.0.{branch}_mlp.net.0.proj'
    values = []
    for suffix in ['weight', 'bias']:
        key = prefix+'.'+suffix
        with safe_open(model/index[key], framework='pt', device='cpu') as f:
            values.append(f.get_tensor(key).to(device='cuda', dtype=torch.bfloat16))
    weight, bias = values
    x = torch.randn((tokens, weight.shape[1]), device='cuda', dtype=torch.bfloat16)
    def eager():
        return F.gelu(F.linear(x, weight, bias), approximate='tanh')
    def fused():
        return torch._addmm_activation(bias, x, weight.t(), use_gelu=True)
    funcs = {'eager': eager, 'fused': fused}
    reference, candidate = eager(), fused()
    diff = candidate.float()-reference.float()
    errors = dict(max_abs=diff.abs().max().item(), rmse=diff.square().mean().sqrt().item(),
                  relative_l2=(diff.norm()/reference.float().norm()).item(), pixel_exact=torch.equal(reference, candidate))
    del reference, candidate, diff
    for _ in range(5):
        eager(); fused()
    torch.cuda.synchronize()
    blocks = []
    for arm in ['eager','fused','fused','eager']*2:
        samples=[]
        for _ in range(10):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); out=funcs[arm](); end.record(); end.synchronize()
            samples.append(start.elapsed_time(end))
            del out
        blocks.append(dict(arm=arm, samples_ms=samples, median_ms=statistics.median(samples)))
    medians={arm:statistics.median([v for b in blocks if b['arm']==arm for v in b['samples_ms']]) for arm in funcs}
    row=dict(branch=branch,tokens=tokens,weight_key=prefix,weight_shape=list(weight.shape),errors=errors,
             medians_ms=medians,eager_over_fused=medians['eager']/medians['fused'],blocks=blocks)
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='blocks'}),flush=True)
result=dict(status='diagnostic_only_not_model_qualification',gpu=torch.cuda.get_device_name(),
            cc=torch.cuda.get_device_capability(),torch=torch.__version__,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
(root/'evidence/preflight/joy-gelu-sm120-probe.json').write_text(json.dumps(result,indent=2)+'\n')
