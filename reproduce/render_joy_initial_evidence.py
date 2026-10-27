"""Lay out unmodified, full-resolution generated PNGs with evidence captions."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
main = 'sgl-joy-gelu-v1-main-high'
candidate = 'sgl-joy-gelu-v1-pr-high'
quality = json.loads((a.root / (candidate + '-quality.json')).read_text())
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path, 27)
small = ImageFont.truetype(font_path, 23)
records = []
for prompt in ['00-edit-short', '01-edit-long', '02-edit-chinese']:
    sources = [a.root / arm / prompt / 'output.png' for arm in (main, candidate)]
    images = [Image.open(source).convert('RGB') for source in sources]
    assert all(im.size == (1024, 1024) for im in images)
    row = next(r for r in quality['results'] if r['prompt'] == prompt and r['reference'].startswith('sgl-'))
    # Paste each complete source at 1:1 pixel scale: no crop, resize, recoloring, or synthetic content.
    canvas = Image.new('RGB', (2128, 1224), '#f8fafc')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), 'JoyAI Image Edit | RTX PRO 6000 | 1024x1024 | 30 steps | seed 1101', font=font, fill='#111827')
    draw.text((24, 55), 'BF16 DiT / encoder / VAE | CFG 4 | high | transformer: cuDNN SDPA', font=small, fill='#334155')
    draw.text((24, 100), 'SGLang main: af21e2b989e4', font=font, fill='#111827')
    draw.text((1080, 100), 'Candidate v1: image GELU epilogue (uncommitted)', font=font, fill='#111827')
    for x, im in zip((24, 1080), images):
        canvas.paste(im, (x, 148))
        assert canvas.crop((x, 148, x + 1024, 1172)).tobytes() == im.tobytes()
    draw.text((24, 1185), f'{prompt} | Main vs candidate: SSIM {row["ssim"]:.6f}; PSNR {row["psnr_db"]:.4f} dB', font=small, fill='#334155')
    target = a.output / (prompt + '-main-vs-pr.png')
    canvas.save(target)
    records.append(dict(prompt=prompt, artifact=target.name,
                        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        source_paths=[str(s) for s in sources],
                        source_sha256=[hashlib.sha256(s.read_bytes()).hexdigest() for s in sources],
                        composition='Full 1024x1024 source PNG pixels pasted unchanged, labels outside image bounds.',
                        ssim=row['ssim'], psnr_db=row['psnr_db']))
(a.output / 'media-manifest.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2))
