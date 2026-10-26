"""Evaluate archived real MOSS serving outputs against frozen eager references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

p=argparse.ArgumentParser()
p.add_argument('candidate')
p.add_argument('--reference', action='append', default=[])
p.add_argument('--root',type=Path)
p.add_argument('--fail-on-quality',action='store_true')
p.add_argument('--require-pixel-identical',action='store_true')
a=p.parse_args()
root=a.root or Path(__file__).resolve().parents[1]/'evidence/qualification'
rows=[]
for path in sorted((root/a.candidate).glob('*/output.png')):
 candidate=np.asarray(Image.open(path).convert('RGB'))
 for ref in a.reference or ['sgl-moss-eager','omni-moss-eager']:
  other=root/ref/path.parent.name/'output.png'
  baseline=np.asarray(Image.open(other).convert('RGB'))
  same=bool(np.array_equal(candidate,baseline))
  ssim=float(structural_similarity(baseline,candidate,channel_axis=2,data_range=255))
  psnr=float(peak_signal_noise_ratio(baseline,candidate,data_range=255)) if not same else None
  rows.append(dict(prompt=path.parent.name,reference=ref,pixel_identical=same,ssim=ssim,psnr_db=psnr,psnr_infinite=same,
                   passes_cross_framework_gate=same or (ssim>=.95 and psnr>=28),
                   candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(other.read_bytes()).hexdigest()))
assert len(rows)>=4
result=dict(candidate=a.candidate,thresholds=dict(ssim=.95,psnr_db=28),results=rows,all_pass=all(r['passes_cross_framework_gate'] for r in rows))
(root/(a.candidate+'-quality.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

if a.fail_on_quality and (not result['all_pass'] or (a.require_pixel_identical and not all(r['pixel_identical'] for r in rows))):
 raise SystemExit('Quality gate failed; preserve evidence and do not benchmark this configuration.')
