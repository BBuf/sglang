"""Audit native BCG replay, quality changes and unchanged real model outputs."""
import hashlib,json,re
from pathlib import Path
R=Path('/task');run='anima-cudnn-bcg-audit1';rows=[]
for index,arm in enumerate(['main','candidate']):
 name=f'{run}-{index:02d}-{arm}'
 folder=R/'block-runs'/name
 meta=json.loads((folder/'metadata.json').read_text());assert meta['status']=='complete'
 files=list((R/'outputs/diagnostics'/f'{run}-{arm}').glob('*.jsonl'));assert len(files)==1
 records=[json.loads(x) for x in files[0].read_text().splitlines()];assert len(records)==18
 warm=[x for x in records if x['is_warmup']];serving=[x for x in records if not x['is_warmup']];assert len(warm)==1 and len(serving)==17
 captures=re.findall(r'\[Diffusion BCG\] captured (\d+) segment\(s\)',(folder/'server.log').read_text());assert captures==['57']
 assert all(x['capture_calls']==0 for x in serving)
 assert all(x['guidance_scale']==4 and (x['width'],x['height'])==(1024,1024) for x in serving)
 assert [x['quality'] for x in serving]==['lossless']*4+['exact']*4+['high']*4+['lossless']*5
 for row in serving:
  expected_replay=100 if row['quality']=='lossless' else 0
  assert row['replay_calls']==expected_replay and row['python_model_forwards']==100-expected_replay
  assert row['rope_gate']['disabled']==(arm=='main') and row['rope_gate']['verified']==(arm=='candidate')
  runners={x['quality']:x['entries'] for x in row['runners']};assert runners['lossless']==1
  assert all(v==0 for k,v in runners.items() if k!='lossless')
 originals=R/'outputs/qualification/sgl-anima-anima-backends-screen1-04-main-cudnn-lossless'
 labels=[f'sgl-anima-{name}-lossless'+suffix for suffix in ['', '-switch-exact','-switch-high','-switch-lossless','-profile']]
 images=[]
 for label in labels:
  d=R/'outputs/qualification'/label
  result=json.loads((d/'results.json').read_text());assert len(result)==(1 if label.endswith('-profile') else 4)
  for value in result:
   image=d/Path(value['request']).name/'output.png';reference=originals/image.parent.name/'output.png'
   assert image.read_bytes()==reference.read_bytes()
   assert hashlib.sha256(image.read_bytes()).hexdigest()==value['sha256']
   perf=json.loads((image.parent/'perf.json').read_text());record=next(x for x in serving if x['request_id']==perf['request_id'])
   images.append(dict(label=label,sha256=value['sha256'],request_id=record['request_id'],quality=record['quality'],replay_calls=record['replay_calls'],capture_calls=record['capture_calls']))
 rows.append(dict(arm=arm,source_sha=meta['source_sha'],source_diff_sha256=meta['source_diff_sha256'],observer_sha256=records[0]['hook_sha256'],actual_warmup_captures=1,segments=57,warmup_capture_method_calls=warm[0]['capture_calls'],serving_requests=17,serving_capture_method_calls=0,serving_replay_calls=sum(x['replay_calls'] for x in serving),quality_switches=['lossless','exact','high','lossless'],all_images_exact_corresponding_native_main=True,images=images))
result=dict(status='pass',rows=rows,new_bcg_support_claim=False,scope='Native 1024x1024,50steps,CFG4,batch1,short/long/Chinese/repeat,512-token conditioner output. Existing BCG support; exact/high intentionally eager, lossless return reuses graph. This is instrumented V1 compatibility, not final-head performance.',capture_note='capture_calls counts calls to idempotent capture(), not graph creation. Warmup calls100 but exactly1 actual graph; no serving capture calls.')
out=R/'evidence/preflight/anima-cudnn-bcg-audit1-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
