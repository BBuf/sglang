"""Compare saved native stage values; do not run either model or alter outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/task')
left = ROOT / 'outputs/diagnostics/qwen21-numerics1-sgl'
right = ROOT / 'outputs/diagnostics/qwen21-numerics1-omni'


def metrics(a, b):
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    result = dict(shape_left=list(a.shape), shape_right=list(b.shape),
                  dtype_left=str(a.dtype), dtype_right=str(b.dtype))
    if a.shape != b.shape:
        return result
    af, bf = a.double(), b.double()
    error = (af - bf).abs()
    result.update(exact=torch.equal(a, b), differing=int(torch.count_nonzero(error)),
                  max_abs=float(error.max()), mean_abs=float(error.mean()),
                  relative_l2=float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(af).clamp_min(1e-30)))
    return result


rows = []
for source in sorted(left.glob('*')):
    peer = right/source.name
    assert peer.is_dir(), source
    meta = json.loads((source/'metadata.json').read_text())
    peer_meta = json.loads((peer/'metadata.json').read_text())
    assert meta['prompt'] == peer_meta['prompt']
    captures = []
    for p in sorted(source.glob('*.pt')):
        if not (peer/p.name).exists():
            captures.append(dict(file=p.name, missing_peer=True))
            continue
        a = torch.load(p, weights_only=True, map_location='cpu')
        b = torch.load(peer/p.name, weights_only=True, map_location='cpu')
        captures.append(dict(file=p.name, tensors={k:metrics(v,b[k]) for k,v in a.items()},
                             source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             peer_sha256=hashlib.sha256((peer/p.name).read_bytes()).hexdigest()))
    rows.append(dict(key=source.name, prompt=meta['prompt'], metadata_left=meta,
                     metadata_right=peer_meta, captures=captures))

output_checks = []
for arm in ['sgl','omni']:
    for task in ['t2i','edit']:
        base=ROOT/'outputs/qualification'/f'{arm}-qwen21-base-{task}-native1'
        new=ROOT/'outputs/qualification'/f'{arm}-qwen21-base-{task}-numerics1'
        for p in sorted(base.glob('*/output.png')):
            q=new/p.parent.name/p.name
            with Image.open(p) as a, Image.open(q) as b:
                equal=a.mode==b.mode=='RGBA' and np.array_equal(np.asarray(a),np.asarray(b))
            output_checks.append(dict(arm=arm, task=task, prompt=p.parent.name,
                                      rgba_pixel_exact_after_instrumentation=equal,
                                      base_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                      diagnostic_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))

result=dict(diagnostic_only=True, stages=rows, output_checks=output_checks,
            all_native_outputs_preserved=all(x['rgba_pixel_exact_after_instrumentation'] for x in output_checks))
out=ROOT/'evidence/preflight/qwen21-native-stage-tensor-comparison.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print('Native RGBA outputs preserved:',result['all_native_outputs_preserved'])
for row in rows:
    print(json.dumps(dict(key=row['key'],prompt=row['prompt'][:35],captures=[x for x in row['captures'] if x['file'] in ['text_encoder.pt','step-00-before.pt','step-00-after.pt']]),ensure_ascii=False))
