"""Spatial-row reuse probe. Numerical/operator screen only, never model admission."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import pynvml
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32
from probe_sana_video_depthwise_glu1 import fused_bias_glu, compare


@triton.jit
def rows_kernel(X, W, Bias, Out, N: tl.constexpr, C: tl.constexpr,
                H: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr,
                BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0,BLOCK)
    ht = tl.cdiv(H, ROWS)
    mask = offsets < N * ht * WIDTH * C
    channel = offsets % C
    pixel = offsets // C
    x = pixel % WIDTH
    y = (pixel // WIDTH) % ht * ROWS
    batch = pixel // (WIDTH * ht)
    hidden0 = tl.full((BLOCK,), 0, tl.float32)
    gate0 = tl.full((BLOCK,), 0, tl.float32)
    hidden1 = tl.full((BLOCK,), 0, tl.float32)
    gate1 = tl.full((BLOCK,), 0, tl.float32)
    hidden2 = tl.full((BLOCK,), 0, tl.float32)
    gate2 = tl.full((BLOCK,), 0, tl.float32)
    hidden3 = tl.full((BLOCK,), 0, tl.float32)
    gate3 = tl.full((BLOCK,), 0, tl.float32)
    for row in tl.static_range(ROWS+2):
        for col in tl.static_range(3):
            sy = y + row - 1
            sx = x + col - 1
            valid = mask & (sy >= 0) & (sy < H) & (sx >= 0) & (sx < WIDTH)
            source = ((batch * H + sy) * WIDTH + sx) * (2*C) + channel
            a = tl.load(X + source, valid, other=0).to(tl.float32)
            b = tl.load(X + source + C, valid, other=0).to(tl.float32)
            if ROWS > 0 and row >= 0 and row < 3:
                tap0 = (row-0)*3+col
                wa0 = tl.load(W+channel*9+tap0, mask, other=0).to(tl.float32)
                wb0 = tl.load(W+(channel+C)*9+tap0, mask, other=0).to(tl.float32)
                hidden0 = tl.fma(a, wa0, hidden0)
                gate0 = tl.fma(b, wb0, gate0)
            if ROWS > 1 and row >= 1 and row < 4:
                tap1 = (row-1)*3+col
                wa1 = tl.load(W+channel*9+tap1, mask, other=0).to(tl.float32)
                wb1 = tl.load(W+(channel+C)*9+tap1, mask, other=0).to(tl.float32)
                hidden1 = tl.fma(a, wa1, hidden1)
                gate1 = tl.fma(b, wb1, gate1)
            if ROWS > 2 and row >= 2 and row < 5:
                tap2 = (row-2)*3+col
                wa2 = tl.load(W+channel*9+tap2, mask, other=0).to(tl.float32)
                wb2 = tl.load(W+(channel+C)*9+tap2, mask, other=0).to(tl.float32)
                hidden2 = tl.fma(a, wa2, hidden2)
                gate2 = tl.fma(b, wb2, gate2)
            if ROWS > 3 and row >= 3 and row < 6:
                tap3 = (row-3)*3+col
                wa3 = tl.load(W+channel*9+tap3, mask, other=0).to(tl.float32)
                wb3 = tl.load(W+(channel+C)*9+tap3, mask, other=0).to(tl.float32)
                hidden3 = tl.fma(a, wa3, hidden3)
                gate3 = tl.fma(b, wb3, gate3)
    bh = tl.load(Bias+channel, mask, other=0).to(tl.float32)
    bg = tl.load(Bias+channel+C, mask, other=0).to(tl.float32)
    if ROWS > 0:
        hidden0 = round_bf16_to_fp32(round_bf16_to_fp32(hidden0)+bh)
        gate0 = round_bf16_to_fp32(round_bf16_to_fp32(gate0)+bg)
        gate0 = round_bf16_to_fp32(gate0 * tl.sigmoid(gate0))
        dest0 = ((batch*H+y+0)*WIDTH+x)*C+channel
        tl.store(Out+dest0, hidden0*gate0, mask & (y+0<H))
    if ROWS > 1:
        hidden1 = round_bf16_to_fp32(round_bf16_to_fp32(hidden1)+bh)
        gate1 = round_bf16_to_fp32(round_bf16_to_fp32(gate1)+bg)
        gate1 = round_bf16_to_fp32(gate1 * tl.sigmoid(gate1))
        dest1 = ((batch*H+y+1)*WIDTH+x)*C+channel
        tl.store(Out+dest1, hidden1*gate1, mask & (y+1<H))
    if ROWS > 2:
        hidden2 = round_bf16_to_fp32(round_bf16_to_fp32(hidden2)+bh)
        gate2 = round_bf16_to_fp32(round_bf16_to_fp32(gate2)+bg)
        gate2 = round_bf16_to_fp32(gate2 * tl.sigmoid(gate2))
        dest2 = ((batch*H+y+2)*WIDTH+x)*C+channel
        tl.store(Out+dest2, hidden2*gate2, mask & (y+2<H))
    if ROWS > 3:
        hidden3 = round_bf16_to_fp32(round_bf16_to_fp32(hidden3)+bh)
        gate3 = round_bf16_to_fp32(round_bf16_to_fp32(gate3)+bg)
        gate3 = round_bf16_to_fp32(gate3 * tl.sigmoid(gate3))
        dest3 = ((batch*H+y+3)*WIDTH+x)*C+channel
        tl.store(Out+dest3, hidden3*gate3, mask & (y+3<H))


def main():
    root = Path('/task/evidence/diagnostics/sana-video-depthwise-glu-rows2')
    previous = json.loads((root.parent/'sana-video-depthwise-glu-screen1/result.json').read_text())
    assert previous['status'] == 'complete'
    pynvml.nvmlInit()
    assert pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(0)) == 'GPU-1565b45e-a702-2015-9776-61f68c36fd59'
    assert all(not pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(2))
    root.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    report = dict(started_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_performance_admitted=False,correctness=[],rows=[])
    def save():
        (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for seed,shape in [(505,(2,192,7,11)),(606,(21,13440,30,52))]:
        n,dc,h,w = shape
        c = dc//2
        generator = torch.Generator(device='cuda:0').manual_seed(seed)
        x = torch.randn((n,h,w,dc), generator=generator, device='cuda', dtype=torch.bfloat16).permute(0,3,1,2)
        weight = torch.randn((dc,1,3,3), generator=generator, device='cuda', dtype=torch.bfloat16)*.1
        bias = torch.randn((dc,), generator=generator, device='cuda', dtype=torch.bfloat16)*.1
        def baseline():
            return fused_bias_glu(F.conv2d(x,weight,None,padding=1,groups=dc),bias)
        reference = baseline()
        configs = [(r,b) for r in (2,4) for b in (128,256,512)]
        def candidate(config):
            rows,block = config
            out = torch.empty_like(reference,memory_format=torch.channels_last)
            kernel = rows_kernel[(triton.cdiv(n*triton.cdiv(h,rows)*w*c,block),)](
                x,weight,bias,out,n,c,h,w,rows,block,num_warps=4,enable_fp_fusion=False)
            return out,kernel
        for config in configs:
            out,kernel = candidate(config)
            comparison = compare(out,reference)
            report['correctness'].append(dict(seed=seed,shape=list(shape),config=list(config),comparison=comparison))
            (root/f'seed{seed}-rows{config[0]}-block{config[1]}.ptx').write_text(kernel.asm['ptx'])
            save()
            assert comparison['exact']
        if seed==505:
            continue
        sequence = [None]+configs+list(reversed(configs))+[None]
        for config in sequence:
            fn = baseline if config is None else lambda: candidate(config)[0]
            for _ in range(6):
                out = fn()
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
            for start,end in zip(starts,ends):
                start.record();out=fn();end.record()
            torch.cuda.synchronize()
            times = [a.elapsed_time(b) for a,b in zip(starts,ends)]
            row = dict(config=config,ms=times,mean_ms=statistics.mean(times),median_ms=statistics.median(times))
            report['rows'].append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k!='ms'}),flush=True)
    report.update(status='complete',finished_unix=time.time());save()


if __name__=='__main__':
    main()
