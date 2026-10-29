"""Publish unchanged pixels from verified final-head native generation runs."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence';run='anima-rope-final-compare1'
heads={'main_eager':'af21e2b989e482cfdce39689126562f2209d53ac','final_eager':'883ad602418f51dc406378e1a5aa0ba8438f6b27','main_bcg':'af21e2b989e482cfdce39689126562f2209d53ac','final_bcg':'883ad602418f51dc406378e1a5aa0ba8438f6b27','omni':'817d4c17feb702289a2d2b2cbacfd594c262b7a7'}
D=E/'media/anima-rope-final';D.mkdir(exist_ok=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',28)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
configs={'main_eager':1,'final_eager':3,'main_bcg':4,'final_bcg':5,'omni':0};sources={}
for config,index in configs.items():
 arm='omni' if config=='omni' else 'sgl';name=f'{run}-{index:02d}-{config}'
 meta=json.loads((E/'block-runs'/name/'metadata.json').read_text());assert meta['status']=='complete' and meta['source_sha']==heads[config]
 folder=E/'qualification'/f'{arm}-anima-{name}-lossless';rows=json.loads((folder/'results.json').read_text())
 assert len(rows)==4 and rows[0]['sha256']==rows[3]['sha256']
 sources[config]=(folder,rows)
panels=[]
for label,left,right in [('eager-main-vs-pr','main_eager','final_eager'),('bcg-main-vs-pr','main_bcg','final_bcg'),('pr-vs-omni','final_eager','omni')]:
 size=1024;gap=24;top=120;row_label=46;canvas=Image.new('RGB',(size*2+gap*3,top+3*(size+row_label)+gap),(248,249,251));draw=ImageDraw.Draw(canvas)
 draw.text((gap,12),'Anima Base v1.0 | 1024 x 1024 | BF16 native | seed42 | 50 steps | CFG4',font=font,fill='black')
 draw.text((gap,48),'RTX PRO 6000 Blackwell | native cuDNN SGL | negative prompt empty | batch1',font=small,fill='black')
 draw.text((gap,83),left+' '+heads[left][:12],font=font,fill='black');draw.text((size+gap*2,83),right+' '+heads[right][:12],font=font,fill='black')
 evidence=[]
 for i,name in enumerate(['Short prompt','Long prompt','Chinese prompt (cross-framework quality failed)']):
  y=top+i*(size+row_label);images={};manifest={}
  for column,config in enumerate([left,right]):
   folder,rows=sources[config];row=rows[i];p=folder/Path(row['request']).name/'output.png';assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
   im=Image.open(p);assert im.mode=='RGB' and im.size==(size,size);images[config]=np.array(im)
   x=gap+column*(size+gap);draw.text((x,y),name+' | '+row['sha256'][:10],font=small,fill='black');canvas.paste(im,(x,y+row_label))
   assert np.array_equal(np.array(canvas)[y+row_label:y+row_label+size,x:x+size],images[config])
   manifest[config]=dict(path=str(p.relative_to(ROOT)),sha256=row['sha256'],source_sha=heads[config])
  equal=np.array_equal(images[left],images[right])
  if right!='omni':assert equal
  evidence.append(dict(prompt_index=i,pixel_exact=bool(equal),sources=manifest))
 p=D/(label+'.png');canvas.save(p);panels.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_pixels_unchanged=True,rows=evidence))
(D/'manifest.json').write_text(json.dumps(dict(actual_generated_outputs=True,head=heads['final_eager'],panels=panels,visual_review_pending=True),indent=2)+'\n')
print(json.dumps([x['path'] for x in panels]))
