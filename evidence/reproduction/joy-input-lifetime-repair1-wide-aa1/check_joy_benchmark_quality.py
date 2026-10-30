"""Check every actual measured/edit warmup image; native Omni fixed-seed outputs can vary."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--reference',required=True,type=Path);a=p.parse_args()
root=Path('/task/outputs/benchmarks')/a.run_id
reference=np.asarray(Image.open(a.reference).convert('RGB'))
rows=[]
for row in json.loads((root/'results.json').read_text()):
 path=root/f'{row["index"]:02d}'/'output.png'
 image=np.asarray(Image.open(path).convert('RGB'))
 assert image.shape==reference.shape
 exact=bool(np.array_equal(image,reference))
 ssim=float(structural_similarity(reference,image,data_range=255,channel_axis=2))
 psnr=None if exact else float(peak_signal_noise_ratio(reference,image,data_range=255))
 rows.append(dict(index=row['index'],warmup=row['warmup'],candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pixel_exact=exact,ssim=ssim,psnr_db=psnr,pass_gate=exact or(ssim>=.95 and psnr>=28)))
result=dict(reference=str(a.reference),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),rows=rows,all_pass=all(r['pass_gate'] for r in rows))
(root/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['all_pass'],'Quality failed; retain and invalidate performance'
