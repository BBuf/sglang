"""Audit complete primary A/A requests, pixels, source and sampled GPU ownership."""
import base64,hashlib,json,statistics
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];RUN='qwen21-gate-up-main-high-aa1'
rows=[];manifest=[]
for i in [0,1]:
 block=f'{RUN}-{i:02d}-main';bm=json.loads((ROOT/'evidence/block-runs'/block/'metadata.json').read_text())
 assert bm['status']=='complete' and bm['source_sha']=='af21e2b989e482cfdce39689126562f2209d53ac' and bm['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
 for task in ['t2i','edit']:
  d=ROOT/'evidence/benchmarks'/f'{block}-{task}';m=json.loads((d/'metadata.json').read_text());r=json.loads((d/'results.json').read_text());assert len(r)==6 and not m['profiler']
  q=ROOT/'evidence/qualification'/f'sgl-qwen21-base-{task}-{block}-high';qualified=json.loads((q/'results.json').read_text())[0]['sha256']
  c=ROOT/'inputs'/f'qwen21-base-{task}-workloads.json';contract=json.loads(c.read_text());assert m['input_sha256']==hashlib.sha256(c.read_bytes()).hexdigest()
  allowed=set(m['args']['allowed_gpu_pids']);assert allowed==set(bm['gpu_container_pids'][0])
  for index,row in enumerate(r):
   p=d/f'{index:02d}';request=json.loads((p/'request.json').read_text());resp=json.loads((p/'response.json').read_text());png=(p/'output.png').read_bytes()
   assert row['warmup']==(index<2) and hashlib.sha256(png).hexdigest()==row['sha256']==qualified
   assert base64.b64decode(resp['data'][0]['b64_json'],validate=True)==png
   with Image.open(p/'output.png') as im:assert im.mode=='RGBA' and im.size==(1024,1024)
   for k,v in dict(seed=1101,num_inference_steps=50,n=1,output_compression=0).items():assert int(request[k])==v
   assert float(request['true_cfg_scale'])==1 and request['generator_device']=='cpu' and request['quality']=='high' and request['prompt']==contract['prompts'][0]['prompt']
   samples=json.loads((p/'gpu-samples.json').read_text());assert samples
   for s in samples:
    assert set(s['gpus'][0]['pids'])<=allowed and not s['gpus'][1]['pids']
   rows.append(dict(block=block,task=task,index=index,warmup=row['warmup'],http_seconds=row['http_seconds'],sha256=row['sha256'],gpu_samples=len(samples)))
  for p in sorted(d.rglob('*')):
   if p.is_file():manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
checks={}
for task in ['t2i','edit']:
 vals=[[r['http_seconds'] for r in rows if r['task']==task and r['block'].endswith(f'{i:02d}-main') and not r['warmup']] for i in [0,1]]
 med=[statistics.median(v) for v in vals];rel=max(med)/min(med)-1;cv=max(statistics.stdev(v)/statistics.mean(v)*100 for v in vals)
 checks[task]=dict(medians=med,process_relative_difference=rel,max_cv_percent=cv,pass_gate=rel<=.005 and cv<=.5)
assert all(v['pass_gate'] for v in checks.values())
out=ROOT/'evidence/preflight/qwen21-gate-up-main-high-aa1-audit.json';assert not out.exists();out.write_text(json.dumps(dict(status='pass',raw_output_count=len(rows),primary_output_count=sum(not r['warmup'] for r in rows),source_and_gpu_ownership_pass=True,all_outputs_match_same_process_qualified_short=True,checks=checks,rows=rows,manifest=manifest,denoise_caveat='Primary uses sync-stage-profiling0. perf.json stage/step CPU timing is not synchronized GPU denoise. Separate synced diagnostics/profiles required for denoise claims.'),indent=2)+'\n');print(json.dumps(checks))
