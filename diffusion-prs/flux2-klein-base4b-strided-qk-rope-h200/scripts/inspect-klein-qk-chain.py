"""Attribute the QK chain to the twenty native single-stream attention scopes."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

with contextlib.redirect_stdout(io.StringIO()):
 ns=runpy.run_path('/campaign/inspect-flux2-klein-trace.py')
scopes=[s for name,ss in ns['scopes'].items() if name.startswith('nn.Module: Flux2ParallelSelfAttention_') for s in ss]
assert len(scopes)==20,len(scopes)
rows=[]
for s in scopes:
 kernels=ns['owned'](s)
 chain=[k for k in kernels if any(t in k['name'] for t in ['_flux2_strided_qknorm_rope_kernel','_rms_norm_tiled_onepass','BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel','arange_cuda_out']) or ('direct_copy_kernel_cuda' in k['name'] and 'BFloat16' in k['name'])]
 names=ns['summary'](chain)
 assert len(chain) in (1,6),(s['name'],names)
 rows.append(dict(scope=s['name'],kernels=len(chain),gpu_ms=sum(k['dur'] for k in chain)/1000,chain=names))
report=dict(scopes=len(rows),kernels=sum(r['kernels'] for r in rows),gpu_ms=sum(r['gpu_ms'] for r in rows),rows=rows)
Path(sys.argv[1]).with_name('qk-chain-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
