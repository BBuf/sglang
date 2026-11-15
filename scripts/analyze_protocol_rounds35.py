"""Retain every final-configuration round; bootstrap rounds, never requests."""
import datetime
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

root=Path('/task')
model=sys.argv[1]
assert model in ('nomas','qwen21')
run_receipts={}
heads=dict(main='0745bfe49b855ae48e0c1ef3305362b8e0c78a03',
    PR='37999f7ab31809bd8a9f3cabee3fa50c54f179d6' if model=='nomas' else '15b28203ba81c96fac1b3692d7bc67bac01d8ff5',
    omni='aa2968e858a718989332a61ad40758024cb4a3a5')

def read(path):
    data=path.read_bytes()
    run_receipts[str(path.relative_to(root))]=hashlib.sha256(data).hexdigest()
    return json.loads(data)

def quantile(values,q):
    s=sorted(values);p=(len(s)-1)*q;a=math.floor(p);b=math.ceil(p)
    return s[a]+(s[b]-s[a])*(p-a)

def describe(values):
    return dict(n=len(values),p50_seconds=statistics.median(values),p95_seconds=quantile(values,.95),
        mean_seconds=statistics.mean(values),raw_seconds=values,
        full_range_pct=100*(max(values)/min(values)-1))

def visit(name,arm):
    folder=root/'runs'/name
    state=read(folder/'status.json');plan=read(folder/'visit.json');rows=read(folder/'outputs/results.json')
    assert state['status']=='raw_complete' and state['gpu_cleanup']=='idle_verified'
    assert plan['physical_gpus']==[0] and all(r['matches_reference'] for r in rows)
    assert not any(r.get('profiler') for r in rows)
    assert plan.get('sync_stage_profiling',0)==0 and not plan.get('synchronized_stage_profiling',False)
    assert plan.get('source_sha',state.get('head_sha'))==heads[arm]
    expected=8 if model=='nomas' else 9
    assert len(rows)==expected and [r['role'] for r in rows]==['warmup']*2+['measure']*(expected-2)
    if model=='qwen21':
        assert plan['actual_png_compress_level']==0 and plan['output_compression']==dict(sgl=0,omni=100)
        assert all(r['native_steps_observed']==50 for r in rows)
    else:
        assert plan['poll_seconds']==.02
    measurements=rows[2:]
    memory=[]
    for row in measurements:
        if arm!='omni':
            data=read(Path(row['output']).parent/'perf.json')
            assert data['commit_hash']==heads[arm]
            memory.append(data['memory_checkpoints'])
    samples=[json.loads(line) for line in (folder/'gpu-samples.jsonl').read_text().splitlines()]
    return dict(run=name,arm=arm,head_sha=heads[arm],
        prompts={p:[r['saved_request_e2e_seconds'] for r in measurements if r['prompt_id']==p] for p in sorted({r['prompt_id'] for r in measurements})},
        all_outputs_exact=True,launch_to_ready_seconds=state['launch_to_ready_seconds'],
        warmup_seconds=[r['saved_request_e2e_seconds'] for r in rows[:2]],
        serial_requests_per_second=len(measurements)/((measurements[-1]['timer_ns']['materialized']-measurements[0]['timer_ns']['start'])/1e9),
        native_memory_checkpoints=memory,
        sampled_device_peak_MiB=max(float(r['gpu'].splitlines()[0].split(', ')[2]) for r in samples))

suffix='33' if model=='nomas' else '34'
assert read(root/f'evidence/preflight/{model}-repeat{suffix}b-controller.json')['status']=='complete'
queue=read(root/f'inputs/{model}-repeat{suffix}-queue.json')['visits']
visits={}
rounds={1:[]}
if model=='nomas':
    for tag in ['A1','B1','B2','A2']:
        name='nomas-49-mean-final12-'+tag;arm='main' if tag.startswith('A') else 'PR'
        visits[name]=visit(name,arm);rounds[1].append(name)
else:
    for arm,index in [('omni',1),('main',1),('PR',1),('PR',2),('main',2),('omni',2)]:
        name=f'qwen21-codec0-{arm}-paired-{index:02d}'
        visits[name]=visit(name,arm);rounds[1].append(name)
aa={}
for row in queue:
    name=row['run'];arm=row.get('mode') or ('main' if row['tag'].startswith('A') else 'PR')
    visits[name]=visit(name,arm)
    (aa.setdefault(arm,[]) if row['phase']=='AA' else rounds.setdefault(row['round'],[])).append(name)
assert sorted(rounds)==[1,2,3]
noise={}
for arm,names in aa.items():
    assert len(names)==2
    noise[arm]={}
    for prompt in visits[names[0]]['prompts']:
        a,b=[visits[n]['prompts'][prompt] for n in names]
        bound=max(abs(math.log(statistics.median(a)/statistics.median(b))),math.log(max(a)/min(a)),math.log(max(b)/min(b)))
        noise[arm][prompt]=dict(visits=names,observed_noise_pct=100*math.expm1(bound),samples=[a,b])
comparisons={}
for ref in (['main'] if model=='nomas' else ['main','omni']):
    cells={}
    first=visits[rounds[1][0]]
    for prompt in first['prompts']:
        per_round=[];all_ref=[];all_PR=[]
        for number,names in rounds.items():
            a=[visits[n] for n in names if visits[n]['arm']==ref]
            b=[visits[n] for n in names if visits[n]['arm']=='PR']
            assert len(a)==len(b)==2
            av=[x for v in a for x in v['prompts'][prompt]];bv=[x for v in b for x in v['prompts'][prompt]]
            ratios=[x/y for x,y in zip(av,bv,strict=True)]
            per_round.append(dict(round=number,reference_runs=[v['run'] for v in a],PR_runs=[v['run'] for v in b],
                paired_ratios=ratios,paired_geometric_speedup=statistics.geometric_mean(ratios),
                reference=describe(av),PR=describe(bv)))
            all_ref.extend(av);all_PR.extend(bv)
        log_round=[math.log(r['paired_geometric_speedup']) for r in per_round]
        bootstrap=[math.exp(statistics.mean(sample)) for sample in itertools.product(log_round,repeat=3)]
        speedup=math.exp(statistics.mean(log_round))
        observed=max(noise[ref][prompt]['observed_noise_pct'],describe(all_ref)['full_range_pct'],describe(all_PR)['full_range_pct'])
        cells[prompt]=dict(rounds=per_round,independent_rounds=3,process_visits_per_arm=6,
            reference=describe(all_ref),PR=describe(all_PR),paired_geometric_speedup=speedup,
            E2E_reduction_pct=100*(1-1/speedup),observed_descriptive_noise_pct=observed,
            gain_exceeds_noise=100*(1-1/speedup)>observed,
            all_round_speedups_above_one=all(r['paired_geometric_speedup']>1 for r in per_round),
            whole_round_bootstrap_speedup_interval95=[quantile(bootstrap,.025),quantile(bootstrap,.975)],
            bootstrap_draws=len(bootstrap),bootstrap_unit='one entire balanced round, only three available units')
    comparisons[ref+'_over_PR']=cells
report=dict(model=model,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),heads=heads,physical_gpus=[0],
    all_original_round1_samples_retained=True,AA=noise,comparisons=comparisons,visits=visits,source_receipts=run_receipts,
    model_qualification_still_requires_public_update=True,
    limitations='Three rounds have limited statistical resolution. P95/ranges are descriptive; whole-round bootstrap does not prove broad model coverage. No sample exclusions. Original separate profile/cleanup/stage/quality evidence remains distinct; no new beneficial BCG scene.')
out=root/f'evidence/benchmarks/{model}-three-rounds35.json'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:{p:{a:r[a] for a in ['paired_geometric_speedup','E2E_reduction_pct','observed_descriptive_noise_pct','gain_exceeds_noise','all_round_speedups_above_one']} for p,r in v.items()} for k,v in comparisons.items()},indent=2))
