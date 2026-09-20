"""Attribute image QK copies, norm/RoPE and joint cats in complete native forwards."""
import bisect
import collections
import gzip
import json
from pathlib import Path
import time

root = Path('/campaign')
done = root / 'validate-joy-outplace-v1.exit'
while not done.exists():
    time.sleep(5)
assert done.read_text().strip() == '0'
for arm, label in [('baseline', 'baseline-profile-retry1'), ('candidate', 'outplace-v1-candidate-profile')]:
    folder = root / 'artifacts/joy-image-edit' / label / 'traces'
    with gzip.open(folder / 'forward3.trace.json.gz', 'rt') as stream:
        events = json.load(stream)['traceEvents']
    launches = sorted((e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')
        and 'correlation' in e.get('args', {}) and 'LaunchKernel' in e.get('name', '')), key=lambda e:e['ts'])
    times = [e['ts'] for e in launches]
    kernels = [e for e in events if e.get('cat') == 'kernel']
    by_correlation = collections.defaultdict(list)
    for kernel in kernels:
        by_correlation[kernel['args']['correlation']].append(kernel)
    def owned(scope):
        lo = bisect.bisect_left(times, scope['ts'])
        hi = bisect.bisect_left(times, scope['ts']+scope['dur'])
        return [k for launch in launches[lo:hi] for k in by_correlation[launch['args']['correlation']]]
    groups = {}
    for branch, tokens in [('image',8048), ('text',1004)]:
        scopes = [e for e in events if e.get('name') == 'aten::contiguous'
            and e.get('args',{}).get('Input Dims',[None])[0] == [1,tokens,32,128]]
        ks = [k for scope in scopes for k in owned(scope)]
        expected = 80 if branch == 'text' or arm == 'baseline' else 0
        assert len(scopes) == len(ks) == expected, (arm,branch,len(scopes),len(ks))
        groups[f'{branch}_qk_copies'] = dict(scopes=len(scopes), kernels=len(ks), gpu_ms=sum(k['dur'] for k in ks)/1000)
    for name, predicate, count in [
        ('image_qk_norm_rope', lambda k:'fused_qknorm_rope_warp' in k['name'],40),
        ('qkv_cats', lambda k:'CatArrayBatchedCopy' in k['name'] and 'OpaqueType<2u>' in k['name'],120 if arm=='baseline' else 0),
        ('joint_qkv_copy', lambda k:'_joint_qkv_cat_kernel' in k['name'],0 if arm=='baseline' else 40),
    ]:
        ks=[k for k in kernels if predicate(k)]
        assert len(ks)==count,(arm,name,len(ks))
        groups[name]=dict(kernels=len(ks),gpu_ms=sum(k['dur'] for k in ks)/1000)
    selected=[groups[name] for name in ['image_qk_copies','image_qk_norm_rope','qkv_cats','joint_qkv_copy']]
    result=dict(arm=arm,groups=groups,optimized_chain=dict(kernels=sum(x['kernels'] for x in selected),gpu_ms=sum(x['gpu_ms'] for x in selected)),
        note='One complete third native forward. Copy scopes matched by input shape and CUDA launch correlation. Text copies and two unrelated FP32 concatenations remain unchanged. Position arange is unchanged and excluded from the optimized-chain subtotal.')
    (folder/'copy-qk-chain-evidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
