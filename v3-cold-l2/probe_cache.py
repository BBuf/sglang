import json, statistics
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    props=torch.cuda.get_device_properties(0)
    x=torch.ones(4*1024*1024,device='cuda')  # 16 MiB, comfortably inside L2
    eviction=torch.zeros(5*props.L2_cache_size,device='cuda',dtype=torch.uint8)
    result=dict(l2_bytes=props.L2_cache_size,eviction_bytes=eviction.numel(),input_bytes=x.nbytes)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for cold in (False,True,True,False):
            graph=torch.cuda.CUDAGraph()
            start=torch.cuda.Event(enable_timing=True,external=True)
            end=torch.cuda.Event(enable_timing=True,external=True)
            with torch.cuda.graph(graph,stream=stream):
                x.add_(0)
                if cold: eviction.add_(1)
                start.record(stream)
                out=x+1
                end.record(stream)
            for _ in range(20): graph.replay()
            samples=[]
            for _ in range(100):
                graph.replay();stream.synchronize()
                samples.append(start.elapsed_time(end)*1000)
            assert torch.equal(out,torch.full_like(out,2))
            result.setdefault('cold_us' if cold else 'hot_us',[]).append(statistics.median(samples))
    result['cold_over_hot']=statistics.mean(result['cold_us'])/statistics.mean(result['hot_us'])
    Path('/scratch/cold-l2/results/cache-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
