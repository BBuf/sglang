"""Compare the two SM120 forward tiles on one BF16 packed self-attention input."""
import argparse
import statistics
import torch
from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--lengths',nargs=2,type=int,default=[6177,31])
p.add_argument('--max-length',type=int)
p.add_argument('--tile',type=int,choices=[64,128],help='Profile just one tile')
a=p.parse_args();length=a.max_length or max(a.lengths)
assert length>=max(a.lengths) and min(a.lengths)>=0
assert torch.cuda.get_device_capability()==(12,0)
torch.manual_seed(901103)
with torch.inference_mode():
    q,k,v=torch.randn(sum(a.lengths),3,56,128,device='cuda',dtype=torch.bfloat16).unbind(1)
    cu=torch.tensor([0,a.lengths[0],sum(a.lengths)],device='cuda',dtype=torch.int32)
    expected=None
    for tile in ([a.tile] if a.tile else [64,128,128,64]):
        def forward():
            return _flash_attn_fwd(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=length,max_seqlen_k=length,causal=False,num_splits=1,tile_mn=(tile,64),return_lse=True)
        for _ in range(3):out,lse=forward()
        if expected is None:expected=(out.clone(),lse.clone())
        else:assert torch.equal(out,expected[0]) and torch.equal(lse,expected[1])
        events=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
        events[0].record()
        for end in events[1:]:
            for _ in range(4):forward()
            end.record()
        events[-1].synchronize()
        values=[x.elapsed_time(y)/4 for x,y in zip(events,events[1:])]
        print(f'M{tile} N64, median {statistics.median(values):.5f} ms, samples {values}')
