"""Audit every planned Anima response and GPU sample before paired statistics."""
import argparse
import hashlib
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('plan',type=Path)
p.add_argument('--root',type=Path,default=Path('/task'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
R=a.root
plan=json.loads(a.plan.read_text())
def read(path):return json.loads(path.read_text())
@lru_cache(None)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(xs,q):
 xs=sorted(xs);pos=(len(xs)-1)*q;lo=int(pos);hi=min(lo+1,len(xs)-1)
 return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
contract=read(R/plan['contract'])
assert digest(R/plan['contract'])==plan['sha256'][plan['contract']]
uuids=['GPU-1565b45e-a702-2015-9776-61f68c36fd59','GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd']
EMPTY=hashlib.sha256(b'').hexdigest()
blocks=[]
failed_quality=[]
for index,config in enumerate(plan['order']):
 spec=plan['configurations'][config];arm=spec['arm'];name=f'{plan["id"]}-{index:02d}-{config}'
 server=read(R/'block-runs'/name/'metadata.json')
 assert server['status']=='complete' and server['plan']==plan and server['config']==config
 assert digest(R/'block-runs'/name/'launch.sh')==plan['sha256']['scripts/'+spec['launcher']]
 expected_sha=spec.get('head',contract['pins'][arm]);expected_diff=spec.get('diff_sha256',EMPTY)
 assert server['source_sha']==expected_sha and server['source_diff_sha256']==expected_diff
 assert digest(R/'block-runs'/name/'source.patch')==expected_diff
 source_log=(R/'block-runs'/name/'server.log').read_text()
 if arm=='sgl':assert not any(x in source_log for x in ['Falling back to diffusers backend','Using diffusers backend','Loaded diffusers pipeline'])
 qname=f'{arm}-anima-{name}-{plan["quality"]}'
 qdir=R/'outputs/qualification'/qname
 assert (qdir/'contract.json').read_bytes()==(R/plan['contract']).read_bytes()
 qmeta=read(qdir/'metadata.json');assert qmeta['source_sha']==expected_sha and qmeta['source_diff_sha256']==expected_diff and not qmeta['profiled']
 qrows=read(qdir/'results.json');assert len(qrows)==4 and qrows[0]['sha256']==qrows[-1]['sha256']
 gate=read(R/'outputs/qualification'/(qname+'-quality.json'))
 assert len(gate['results'])==4*len(spec['references'])
 assert {x['reference'] for x in gate['results']}==set(spec['references'])
 for row in qrows:assert digest(qdir/Path(row['request']).name/'output.png')==row['sha256']
 for row in gate['results']:
  assert digest(qdir/row['prompt']/'output.png')==row['candidate_sha256']
  assert digest(R/'outputs/qualification'/row['reference']/row['prompt']/'output.png')==row['reference_sha256']
  if not row['passes_cross_framework_gate']:failed_quality.append(dict(block=name,**row))
 if spec.get('exact_reference'):
  exact=[x for x in gate['results'] if x['reference']==spec['exact_reference']]
  assert len(exact)==4 and all(x['pixel_identical'] for x in exact)
 for prompt in plan['measured_prompts']:
  gates=[x for x in gate['results'] if x['prompt'].endswith('-'+prompt)]
  assert gates and all(x['passes_cross_framework_gate'] for x in gates)
  qhash=next(x['sha256'] for x in qrows if Path(x['request']).name.endswith('-'+prompt))
  folder=R/'outputs/benchmarks'/(name+'-'+prompt)
  meta=read(folder/'metadata.json');rows=read(folder/'results.json');summary=read(folder/'summary.json')
  assert meta['arm']==arm and meta['source_sha']==expected_sha and meta['source_diff_sha256']==expected_diff
  assert meta['physical_gpus']==[0] and meta['input_sha256']==digest(R/plan['contract'])
  assert meta['args']['prompt_id']==prompt and meta['args']['quality']==plan['quality']
  assert meta['args']['repeats']==plan['repeats'] and meta['args']['warmups']==plan['warmups']
  assert meta['profiler']==plan.get('diagnostic_timing',False)
  assert len(rows)==plan['warmups']+plan['repeats']
  allowed=set(meta['args']['allowed_gpu_pids']);assert allowed==set(server['gpu_container_pids'][0]) and not server['gpu_container_pids'][1]
  times=[];mem=[];telemetry=[];dispatch=[]
  for i,row in enumerate(rows):
   assert row['index']==i and row['warmup']==(i<plan['warmups']) and row['sha256']==qhash
   d=folder/f'{i:02d}';assert digest(d/'output.png')==qhash
   request=read(d/'request.json')
   for key in ['seed','guidance_scale','generator_device','num_inference_steps','negative_prompt']:
    assert request[key]==contract[key]
   assert request['prompt']==next(x['prompt'] for x in contract['prompts'] if x['id']==prompt)
   assert request['model']==contract[arm+'_model_path'] and request['n']==1
   assert request['size']==f'{contract["width"]}x{contract["height"]}'
   assert request['output_format']=='png' and request['response_format']=='b64_json'
   assert request['output_compression']==contract['png_compression'][arm+'_output_compression']
   if arm=='sgl':assert request['max_sequence_length']==512 and request['quality']==plan['quality'] and not request['profile']
   else:assert 'max_sequence_length' not in request
   samples=read(d/'gpu-samples.json');assert len(samples)>=2
   for sample in samples:
    g0,g1=sample['gpus'];assert [g0['uuid'],g1['uuid']]==uuids
    assert set(g0['pids'])<=allowed and not g1['pids']
   if row['warmup']:continue
   times.append(row['http_seconds']);mem.append(max(x['gpus'][0]['memory_bytes'] for x in samples));telemetry.extend(x['gpus'][0] for x in samples)
   if arm=='sgl':
    perf=read(d/'perf.json');assert len(perf['denoise_steps_ms'])==contract['num_inference_steps']
    dispatch.append(next(x['duration_ms']/1000 for x in perf['steps'] if x['name']=='DenoisingStage'))
  assert summary['p50_seconds']==statistics.median(times)
  half=len(times)//2
  blocks.append(dict(index=index,config=config,prompt=prompt,name=name,source_sha=expected_sha,source_diff_sha256=expected_diff,
    http_seconds=times,p50_seconds=statistics.median(times),p95_seconds=percentile(times,.95),cv_percent=100*statistics.stdev(times)/statistics.mean(times),
    half_median_ratio=statistics.median(times[:half])/statistics.median(times[half:]),
    requests_per_second=len(times)/sum(times),peak_gpu_memory_bytes=max(mem),launch_to_ready_seconds=server['launch_to_ready_seconds'],
    denoise_host_dispatch_seconds=dispatch,denoise_is_synchronized=plan.get('diagnostic_timing',False),
    sm_clock_range=[min(x['sm_clock_mhz'] for x in telemetry),max(x['sm_clock_mhz'] for x in telemetry)],
    temperature_range=[min(x['temperature_c'] for x in telemetry),max(x['temperature_c'] for x in telemetry)]))
r=dict(run=plan['id'],status='raw_contract_pass',plan_sha256=digest(a.plan),all_raw_source_inputs_outputs_and_gpu_ownership_pass=True,
 all_samples_retained=True,blocks=blocks,retained_quality_failures=failed_quality,
 caveat='Only frozen English short/long are measured. Chinese failures remain in qualification. SYNC0 denoise host dispatch timings are not synchronized GPU latency. Readiness uses existing disk/compiler caches, not a cold-start claim.')
if 'noise_policy' in plan:
 policy=plan['noise_policy'];r['aa_noise']=[dict(config=x['config'],prompt=x['prompt'],cv_percent=x['cv_percent'],half_median_difference_percent=100*abs(x['half_median_ratio']-1),passed=x['cv_percent']<=policy['cv_percent_max'] and 100*abs(x['half_median_ratio']-1)<=policy['half_median_difference_percent_max']) for x in blocks]
 r['all_aa_noise_pass']=all(x['passed'] for x in r['aa_noise'])
if 'pair_indices' in plan:
 r['paired']={}
 for prompt in plan['measured_prompts']:
  selected={x['index']:x for x in blocks if x['prompt']==prompt};r['paired'][prompt]={}
  for name,pairs in plan['pair_indices'].items():
   ratios=[selected[i]['p50_seconds']/selected[j]['p50_seconds'] for i,j in pairs]
   geometric=math.exp(statistics.mean(math.log(x) for x in ratios))
   r['paired'][prompt][name]=dict(pairs=pairs,ratios=ratios,geometric_mean=geometric,latency_reduction_percent=100*(1-1/geometric),all_pairs_faster=all(x>1 for x in ratios),min_baseline_over_max_candidate=min(t for i,j in pairs for t in selected[i]['http_seconds'])/max(t for i,j in pairs for t in selected[j]['http_seconds']))
a.output.parent.mkdir(parents=True,exist_ok=True)
assert not a.output.exists()
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['blocks','retained_quality_failures']},indent=2))
for row in blocks:print(row['index'],row['config'],row['prompt'],round(row['p50_seconds'],6),'CV',round(row['cv_percent'],4))
