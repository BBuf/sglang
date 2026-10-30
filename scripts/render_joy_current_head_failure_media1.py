"""Render actual audited square outputs without treating failed cleanup as a pass."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
root=Path(__file__).resolve().parents[1];e=root/'evidence';run='joy-current-head-square-primary1'
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(e/'preflight'/(run+'-admission.json'));analysis=read(e/'benchmarks'/(run+'-analysis.json'));p=e/'plans'/(run+'.json');plan=read(p)
assert not a['passed'] and a['primary_passed'] and not a['cleanup_passed']
assert a['all_raw_images_and_ownership_audited'] and analysis['source_and_ownership_audit_pass']
assert a['analysis_sha256']==digest(e/'benchmarks'/(run+'-analysis.json'))
assert a['plan_sha256']==digest(p) and analysis['plan']==plan
v=read(e/'checkpoints'/(run+'-local-verification.json'));assert v['all_member_hashes_verified'] and v['all_members_extracted']
contract=read(root/'inputs/joy-workloads.json');assert digest(root/'inputs/joy-workloads.json')==plan['frozen_input_sha256']
gate_path=e/'qualification'/('sgl-joy-'+run+'-02-candidate_current-quality.json')
archive=read(e/'checkpoints'/(run+'-archive-manifest.json'))
gate_record=next(x for x in archive['members'] if x['path']=='outputs/qualification/'+gate_path.name)
assert digest(gate_path)==gate_record['sha256']
gate=read(gate_path);assert gate['all_pass']
dest=e/'media/joy-current-head-square-cleanup-failed-b08c24e9';dest.mkdir(exist_ok=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
rows=[]
for index in range(3):
 pair=[]
 for block_index,arm in [(0,'main'),(2,'candidate_current')]:
  name=f'{run}-{block_index:02d}-{arm}';meta=read(e/'block-runs'/name/'metadata.json');assert meta['source_sha']==plan['configurations'][arm]['head'] and meta['status']=='complete'
  folder=e/'qualification'/('sgl-joy-'+name);result=read(folder/'results.json')[index];request=Path(result['request']).name;form=read(folder/request/'request.json')['form'];assert form['quality']=='high' and form['prompt']==contract['prompts'][index]['prompt'] and form['seed']==str(contract['seed'])
  path=folder/request/'output.png';assert digest(path)==result['sha256'];im=Image.open(path).convert('RGB');assert im.size==(1024,1024)
  pair.append(dict(arm=arm,head=meta['source_sha'],path=str(path.relative_to(root)),sha256=digest(path),image=im,request=request))
 metrics=next(x for x in gate['results'] if x['prompt']==pair[0]['request'] and x['reference']=='sgl-joy-gelu-v1-main-high')
 assert metrics['reference_sha256']==pair[0]['sha256'] and metrics['candidate_sha256']==pair[1]['sha256']
 exact=pair[0]['image'].tobytes()==pair[1]['image'].tobytes();ssim=metrics['ssim'];psnr=metrics['psnr_db']
 assert exact==metrics['pixel_identical'] and metrics['passes_cross_framework_gate'];assert exact or(ssim>=.95 and psnr>=28)
 canvas=Image.new('RGB',(2128,1240),'#f8fafc');draw=ImageDraw.Draw(canvas)
 draw.text((24,16),'JoyAI Image Edit | 1024x1024 | high | BF16 | 30 steps | CFG 4 | seed 1101',font=font,fill='#111827')
 draw.text((24,55),'Actual square rerun: image quality passed; current-head cleanup performance gate FAILED',font=small,fill='#9f1239')
 for x,item in zip([24,1080],pair):
  label='Frozen main' if item['arm']=='main' else 'Official PR head'
  draw.text((x,98),label+' '+item['head'][:12],font=font,fill='#111827');canvas.paste(item['image'],(x,145));assert canvas.crop((x,145,x+1024,1169)).tobytes()==item['image'].tobytes()
 draw.text((24,1190),f'{pair[0]["request"]} | SSIM {ssim:.6f} | PSNR '+('infinite' if psnr is None else f'{psnr:.4f} dB'),font=small,fill='#334155')
 target=dest/(pair[0]['request']+'-main-vs-official-current.png');canvas.save(target)
 rows.append(dict(panel=str(target.relative_to(root)),panel_sha256=digest(target),prompt=contract['prompts'][index]['prompt'],sources=[{k:v for k,v in x.items() if k!='image'} for x in pair],pixel_exact=exact,ssim=ssim,psnr_db=psnr,composition='Full actual source images pasted 1:1; labels only outside source pixels.'))
manifest=dict(source_sha=a['source_sha'],run=run,passed_cleanup=False,archive_sha256=v['sha256'],admission_sha256=digest(e/'preflight'/(run+'-admission.json')),quality_metrics_source=str(gate_path.relative_to(root)),quality_metrics_sha256=digest(gate_path),rows=rows,visual_review_pending=True)
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'panels':len(rows),'directory':str(dest),'metrics':[{k:r[k] for k in ['ssim','psnr_db']} for r in rows]},indent=2))
