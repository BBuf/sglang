"""Diagnostic operator screen only; this cannot admit a performance PR."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import pynvml
import torch
import torch.nn.functional as F
import triton

from probe_sana_video_depthwise_glu1 import probe_kernel, fused_bias_glu, compare


def main():
    root = Path('/task/evidence/diagnostics/sana-video-depthwise-glu-screen1')
    previous = json.loads((root.parent/'sana-video-depthwise-glu-probe1/result.json').read_text())
    assert previous['status'] == 'complete' and previous['orders_exact_on_all_random_cases'] == [0]
    pynvml.nvmlInit()
    assert pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(0)) == 'GPU-1565b45e-a702-2015-9776-61f68c36fd59'
    assert all(not pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(2))
    root.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    n,c,h,w = 21,6720,30,52
    generator = torch.Generator(device='cuda:0').manual_seed(404)
    x = torch.randn((n,h,w,2*c), generator=generator, device='cuda', dtype=torch.bfloat16).permute(0,3,1,2)
    weight = torch.randn((2*c,1,3,3), generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    bias = torch.randn((2*c,), generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    def baseline():
        return fused_bias_glu(F.conv2d(x,weight,None,padding=1,groups=2*c),bias)
    reference = baseline()
    def candidate(block):
        out = torch.empty_like(reference, memory_format=torch.channels_last)
        kernel = probe_kernel[(triton.cdiv(out.numel(),block),)](
            x,weight,bias,out,out,n,c,h,w,0,False,block,
            num_warps=4,enable_fp_fusion=False)
        return out,kernel
    report = dict(started_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_performance_admitted=False,seed=404,shape=list(x.shape),correctness=[],rows=[])
    for block in [256,512,1024,2048]:
        out,kernel = candidate(block)
        report['correctness'].append(dict(block=block,comparison=compare(out,reference)))
        (root/f'block{block}.ptx').write_text(kernel.asm['ptx'])
    assert all(row['comparison']['exact'] for row in report['correctness'])
    sequence = [0,256,512,1024,2048,2048,1024,512,256,0]
    for block in sequence:
        fn = baseline if block == 0 else lambda: candidate(block)[0]
        for _ in range(6):
            result = fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        for start,end in zip(starts,ends):
            start.record();result=fn();end.record()
        torch.cuda.synchronize()
        times = [a.elapsed_time(b) for a,b in zip(starts,ends)]
        row = dict(block=block,ms=times,mean_ms=statistics.mean(times),median_ms=statistics.median(times))
        report['rows'].append(row)
        (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='ms'}),flush=True)
    report.update(status='complete',finished_unix=time.time())
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
