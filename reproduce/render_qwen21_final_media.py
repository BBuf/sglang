"""Place unchanged actual main/final RGBA pixels side by side; no synthesized media."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('size',type=int,choices=[512,1024])
args=p.parse_args()
size=args.size
run=f'qwen21-gate-up-v4-final-{size}-compare1'
heads={'main':'af21e2b989e482cfdce39689126562f2209d53ac','final':'efcb88a201a41b1fecc516a60df0ca0a34fb6dd5'}
dest=ROOT/'evidence/media'/f'qwen21-gate-up-final-{size}'
dest.mkdir(exist_ok=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20 if size==512 else 28)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15 if size==512 else 21)
panels=[]
for task in ['t2i','edit']:
    sources={}
    for index,config in enumerate(['main','final']):
        name=f'{run}-{index:02d}-{config}'
        metadata=json.loads((ROOT/'evidence/block-runs'/name/'metadata.json').read_text())
        assert metadata['status']=='complete' and metadata['source_sha']==heads[config]
        folder=ROOT/'evidence/qualification'/f'sgl-qwen21-base-{task}-{name}-high'
        results=json.loads((folder/'results.json').read_text())
        assert len(results)==4 and results[0]['sha256']==results[3]['sha256']
        sources[config]=(folder,results)
    gap=24
    top=110
    row_label=46
    canvas=Image.new('RGBA',(size*2+gap*3,top+3*(size+row_label)+gap),(248,249,251,255))
    draw=ImageDraw.Draw(canvas)
    draw.text((gap,12),f'Qwen-Image 2.1 | {task.upper()} | {size} x {size} | BF16 high',font=font,fill='black')
    draw.text((gap,45),'Seed 1101 | 50 steps | CFG 1 | RTX PRO 6000 Blackwell',font=small,fill='black')
    draw.text((gap,75),'MAIN af21e2b989e4',font=font,fill='black')
    draw.text((size+gap*2,75),'PR efcb88a201a4',font=font,fill='black')
    rows=[]
    for index,label in enumerate(['Short prompt','Long prompt','Chinese prompt']):
        y=top+index*(size+row_label)
        images={}
        manifests={}
        for column,config in enumerate(['main','final']):
            folder,results=sources[config]
            row=results[index]
            path=folder/Path(row['request']).name/'output.png'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            image=Image.open(path)
            assert image.mode=='RGBA' and image.size==(size,size)
            images[config]=np.asarray(image).copy()
            x=gap+column*(size+gap)
            draw.text((x,y),label+' | '+row['sha256'][:12],font=small,fill='black')
            canvas.paste(image,(x,y+row_label))
            assert np.array_equal(np.asarray(canvas)[y+row_label:y+row_label+size,x:x+size],images[config])
            manifests[config]=dict(path=str(path.relative_to(ROOT)),sha256=row['sha256'],source_sha=heads[config])
        assert np.array_equal(images['main'],images['final'])
        rows.append(dict(prompt_index=index,pixel_exact_rgba=True,sources=manifests))
    output=dest/f'{task}-{size}-main-vs-pr.png'
    canvas.save(output)
    panels.append(dict(path=str(output.relative_to(ROOT)),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_pixels_unchanged=True,rows=rows))
(dest/'manifest.json').write_text(json.dumps(dict(actual_generated_outputs=True,head=heads['final'],panels=panels,visual_review_pending=True),indent=2)+'\n')
print(json.dumps([x['path'] for x in panels]))
