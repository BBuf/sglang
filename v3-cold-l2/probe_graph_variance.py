import json, os
from pathlib import Path
import torch
from bench_cold_l2_final import cold_bench, clone_args, fingerprints, workloads
with torch.inference_mode():
 torch.manual_seed(20261004)
 eviction=torch.zeros(torch.cuda.get_device_properties(0).L2_cache_size*5,device='cuda',dtype=torch.uint8)
 rows=[]
 for name,fn,args,_ in workloads():
  if name not in ('qknorm_complex_triton/17','silu_mul/17','complex_rope/17','cat_pad/8/6/6'):continue
  expected=fingerprints(fn(*clone_args(args)))
  medians=[cold_bench(fn,args,eviction,expected)['cold_median_us'] for _ in range(12)]
  rows.append(dict(name=name,recapture_medians_us=medians))
  print(rows[-1],flush=True)
 Path(os.environ['OUTPUT']).write_text(json.dumps(rows,indent=2)+'\n')
