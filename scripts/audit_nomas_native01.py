"""CPU-only all-frame and native-input comparison for the two49-frame baselines."""
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity

root=Path('/task');out=root/'evidence/quality/nomas-49-native01';out.mkdir(parents=True,exist_ok=False)
contract=json.loads((root/'inputs/nomas-native-workloads01.json').read_text());gate=contract['quality']['cross_framework']
items=[x for x in contract['requests'] if x['frames']==49];assert len(items)==7
runs={a:root/'runs'/('nomas-49-'+a+'-native01') for a in ['sgl','omni']}
results={a:json.loads((r/'outputs/results.json').read_text()) for a,r in runs.items()}
assert all(len(x)==7 for x in results.values())
(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),benchmark=False,quality_gate=gate,native_inputs=[],pairs=[])
def save():
 (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def tensors(arm,i):
 row=results[arm][i];folder=Path(row['output']).parent
 events=json.loads((folder/'native-audit.json').read_text());event=next(e for e in events if e['event']=='native_inputs')
 return event,torch.load(runs[arm]/'runtime-audit'/event['tensor_file'],map_location='cpu',weights_only=True)
for i,item in enumerate(items):
 left,lt=tensors('omni',i);right,rt=tensors('sgl',i);comparison=dict(id=item['id'],tensors={})
 for name in ['initial_noise','text_embeddings','text_mask']:
  a,b=lt[name],rt[name];same_shape=a.shape==b.shape;exact=same_shape and a.dtype==b.dtype and torch.equal(a,b)
  metrics=dict(same_shape=same_shape,left_shape=list(a.shape),right_shape=list(b.shape),left_dtype=str(a.dtype),right_dtype=str(b.dtype),exact=exact)
  if same_shape:
   delta=a.float()-b.float();metrics.update(max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()),relative_l2=float(delta.norm()/a.float().norm().clamp_min(1e-20)))
  comparison['tensors'][name]=metrics
 comparison['timesteps_equal']=left['timesteps']==right['timesteps'];comparison['sigmas_equal']=left['sigmas']==right['sigmas']
 report['native_inputs'].append(comparison);save()

def panel(arrays,labels,item,frame,path):
 width,height=item['width'],item['height'];canvas=Image.new('RGB',(2*width,height+64),'white');draw=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 except OSError:font=ImageFont.load_default(size=20)
 for i,(array,label) in enumerate(zip(arrays,labels)):
  draw.text((i*width+10,4),label,fill='black',font=font)
  draw.text((i*width+10,32),f'Nomas BF16 | {width}x{height}, {item["frames"]}f, 8steps, seed{item["seed"]} | t={frame/item["fps"]:.4f}s',fill='black',font=font)
  canvas.paste(Image.fromarray(array),(i*width,64))
 canvas.save(path)

def compare(name,left_arm,left_i,right_arm,right_i,exact):
 item=items[left_i];assert item['prompt_id']==items[right_i]['prompt_id'] and item['seed']==items[right_i]['seed']
 paths=[Path(results[a][i]['output']) for a,i in [(left_arm,left_i),(right_arm,right_i)]]
 directory=out/name;directory.mkdir();width,height=item['width'],item['height'];processes=[];decode_logs=[];hashes=[hashlib.sha256(),hashlib.sha256()];frames=[];worst=None
 labels=[a+' frozen main '+('first' if i==0 else 'repeat' if i==6 else item['prompt_id']) for a,i in [(left_arm,left_i),(right_arm,right_i)]]
 artifact_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
 for side,p in zip(['left','right'],paths):
  cmd=['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)];meta=json.loads(subprocess.check_output(cmd));(directory/(side+'-ffprobe.json')).write_text(json.dumps(meta,indent=2)+'\n')
  videos=[s for s in meta['streams'] if s['codec_type']=='video'];assert len(videos)==1;v=videos[0];assert (v['width'],v['height'])==(width,height)
  num,den=map(int,v['avg_frame_rate'].split('/'));assert num/den==item['fps'];assert not any(s['codec_type']=='audio' for s in meta['streams'])
  cmd=['ffmpeg','-v','error','-i',str(p),'-map','0:v:0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];(directory/(side+'-decode-command.json')).write_text(json.dumps(cmd)+'\n')
  decode_log=(directory/(side+'-decode.log')).open('xb');decode_logs.append(decode_log)
  processes.append(subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=decode_log))
 try:
  while True:
   raw=[p.stdout.read(width*height*3) for p in processes]
   if not any(raw):break
   assert all(len(x)==width*height*3 for x in raw),'decoded frame counts differ'
   arrays=[np.frombuffer(x,np.uint8).reshape(height,width,3) for x in raw]
   for h,x in zip(hashes,raw):h.update(x)
   equal=raw[0]==raw[1];error=arrays[0].astype(np.float32)-arrays[1].astype(np.float32);mse=float(np.mean(error*error,dtype=np.float64));score=1.0 if equal else float(structural_similarity(*arrays,data_range=255,channel_axis=2));psnr=None if mse==0 else 10*math.log10(255**2/mse)
   row=dict(frame=len(frames),timestamp=len(frames)/item['fps'],exact=equal,ssim=score,psnr_db=psnr,max_abs=float(np.abs(error).max()),rmse=math.sqrt(mse));frames.append(row)
   if row['frame'] in (0,item['frames']//2,item['frames']-1):panel(arrays,labels,item,row['frame'],directory/f'frame-{row["frame"]:03d}.png')
   if worst is None or score<worst[0]:worst=(score,[x.copy() for x in arrays],row['frame'])
  assert len(frames)==item['frames']
  assert all(p.wait(timeout=30)==0 for p in processes)
 finally:
  for p in processes:
   if p.poll() is None:p.terminate();p.wait(timeout=10)
  for log in decode_logs:log.close()
 panel(worst[1],labels,item,worst[2],directory/'worst.png')
 passed=all(x['exact'] for x in frames) and artifact_hashes[0]==artifact_hashes[1] if exact else all(x['ssim']>=gate['ssim_min'] and (x['psnr_db'] is None or x['psnr_db']>=gate['psnr_min_db']) for x in frames)
 metrics=dict(name=name,paths=[str(p) for p in paths],artifact_sha256=artifact_hashes,artifact_exact=artifact_hashes[0]==artifact_hashes[1],decoded_sha256=[x.hexdigest() for x in hashes],exact_required=exact,frames=frames,numeric_passed=passed,worst_frame=worst[2],visual_review='pending')
 (directory/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
 report['pairs'].append(dict(name=name,numeric_passed=passed,artifact_exact=metrics['artifact_exact'],minimum_ssim=min(x['ssim'] for x in frames),minimum_psnr_db=min((x['psnr_db'] for x in frames if x['psnr_db'] is not None),default=None),worst_frame=worst[2],visual_review='pending'));save()
for arm in ['sgl','omni']:compare(arm+'-own-repeat',arm,0,arm,6,True)
for i in range(6):compare('cross-'+items[i]['id'],'omni',i,'sgl',i,False)
report['common_initial_noise_and_masks']=all(x['tensors']['initial_noise']['exact'] and x['tensors']['text_mask']['exact'] and x['timesteps_equal'] and x['sigmas_equal'] for x in report['native_inputs'])
report['numeric_passed']=report['common_initial_noise_and_masks'] and all(x['numeric_passed'] for x in report['pairs']);report['visual_review']='pending';save();print(json.dumps(dict(numeric_passed=report['numeric_passed'],common_initial_noise_and_masks=report['common_initial_noise_and_masks'],pairs=report['pairs'])))
