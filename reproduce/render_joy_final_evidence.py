"""Compose actual main/final-head PNGs at1:1 pixels with external evidence labels."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p=argparse.ArgumentParser()
p.add_argument('--main',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--main-source',type=Path,required=True)
p.add_argument('--candidate-source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
main_meta=json.loads(a.main_source.read_text());pr_meta=json.loads(a.candidate_source.read_text())
main_sha='af21e2b989e482cfdce39689126562f2209d53ac'
pr_sha='4c135ea92d6b4cbb0ead69bb39976f7dbe995455'
assert main_meta['source_sha']==main_sha and pr_meta['source_sha']==pr_sha
empty=hashlib.sha256(b'').hexdigest()
assert main_meta.get('source_diff_sha256',main_meta.get('diff_sha256'))==empty
assert pr_meta['source_diff_sha256']==empty
contract_bytes=(a.candidate/'contract.json').read_bytes()
assert (a.main/'contract.json').read_bytes()==contract_bytes
assert hashlib.sha256(contract_bytes).hexdigest()==pr_meta['input_sha256']
contract=json.loads(contract_bytes)
gate=json.loads(a.candidate.with_name(a.candidate.name+'-quality.json').read_text())
assert gate['all_pass']
main_rows=json.loads((a.main/'results.json').read_text())
pr_rows=json.loads((a.candidate/'results.json').read_text())
a.output.mkdir(parents=True,exist_ok=False)
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
font=ImageFont.truetype(font_path,27);small=ImageFont.truetype(font_path,23)
records=[]
width,height=contract['width'],contract['height']
for index,prompt in enumerate(contract['prompts']):
    name=f'{index:02d}-{prompt["id"]}'
    paths=[folder/name/'output.png' for folder in (a.main,a.candidate)]
    hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert hashes==[main_rows[index]['sha256'],pr_rows[index]['sha256']]
    row=next(r for r in gate['results'] if r['prompt']==name and r['reference']==a.main.name)
    assert hashes==[row['reference_sha256'],row['candidate_sha256']]
    images=[Image.open(path).convert('RGB') for path in paths]
    assert all(im.size==(width,height) for im in images)
    canvas=Image.new('RGB',(2*width+80,height+212),'#f8fafc');draw=ImageDraw.Draw(canvas)
    draw.text((24,16),f'JoyAI Image Edit | RTX PRO 6000 | {width}x{height} | 30 steps | seed1101',font=font,fill='#111827')
    draw.text((24,55),'BF16 DiT / encoder / VAE | native mixed precision | CFG4 | high',font=small,fill='#334155')
    for x,label,im in zip((24,width+56),(f'Main: {main_sha[:12]}',f'PR: {pr_sha[:12]}'),images):
        draw.text((x,100),label,font=font,fill='#111827')
        canvas.paste(im,(x,148))
        assert canvas.crop((x,148,x+width,148+height)).tobytes()==im.tobytes()
    psnr='infinite' if row['psnr_db'] is None else f'{row["psnr_db"]:.4f}dB'
    draw.text((24,height+174),f'{name} | SSIM {row["ssim"]:.6f} | PSNR {psnr}',font=small,fill='#334155')
    target=a.output/(name+'-main-vs-pr.png');canvas.save(target)
    records.append(dict(prompt_id=name,prompt=prompt['prompt'],artifact=target.name,
        artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_paths=[str(p) for p in paths],
        source_sha256=hashes,source_heads=[main_sha,pr_sha],source_metadata=[str(a.main_source),str(a.candidate_source)],
        contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),ssim=row['ssim'],psnr_db=row['psnr_db'],
        composition='Each complete source image pasted at1:1 with unchanged RGB pixels; labels outside source bounds.'))
(a.output/'media-manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'artifact':r['artifact'],'source_heads':r['source_heads']} for r in records],indent=2))
