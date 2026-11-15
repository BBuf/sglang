"""Analyze completed native final/cleanup visits without dropping any samples."""
import datetime
import json
import math
import statistics
from pathlib import Path

root=Path('/task')
queue=json.loads((root/'inputs/nomas-measurements12-queue.json').read_text())
noise=json.loads((root/'evidence/performance/nomas-main-noise10.json').read_text())
visits={}


def quantile(values,q):
    values=sorted(values);i=(len(values)-1)*q;a=math.floor(i);b=math.ceil(i)
    return values[a]+(values[b]-values[a])*(i-a)


def summarize(values):
    return dict(n=len(values),p50_seconds=statistics.median(values),p95_seconds=quantile(values,.95),
                mean_seconds=statistics.mean(values),minimum_seconds=min(values),maximum_seconds=max(values),
                full_range_pct=100*(max(values)/min(values)-1),raw_seconds=values)


for name in queue['visits']:
    folder=root/'runs'/name
    status=json.loads((folder/'status.json').read_text());plan=json.loads((folder/'visit.json').read_text())
    assert status['status']=='raw_complete' and status['gpu_cleanup']=='idle_verified',name
    assert plan['poll_seconds']==.02 and not plan['cross_framework_performance_admitted']
    rows=json.loads((folder/'outputs/results.json').read_text())
    assert len(rows)==8 and all(x['matches_reference'] and not x['profiler'] for x in rows)
    measurements=[x for x in rows if x['role']=='measure'];assert len(measurements)==6
    stages=[];memory=[]
    for row in measurements:
        perf=json.loads((Path(row['output']).parent/'perf.json').read_text())
        assert perf['commit_hash']==plan['source_sha']
        stages.append({x['name']:x['duration_ms']/1000 for x in perf['steps']})
        memory.append(perf['memory_checkpoints']['after_forward'])
    sample_peak=max(float(line['gpu'].splitlines()[0].split(', ')[2]) for line in
                    map(json.loads,(folder/'gpu-samples.jsonl').read_text().splitlines()))
    visits[name]=dict(source_sha=plan['source_sha'],source_repo=plan['source_repo'],
                      sync_stage_profiling=plan['sync_stage_profiling'],
                      per_prompt={p:summarize([x['saved_request_e2e_seconds'] for x in measurements if x['prompt_id']==p]) for p in ['short','long']},
                      rows=measurements,stage_diagnostics=stages,memory=memory,
                      sampled_total_device_peak_MiB=sample_peak,
                      serial_requests_per_second=6/((measurements[-1]['timer_ns']['materialized']-measurements[0]['timer_ns']['start'])/1e9),
                      launch_to_ready_seconds=status['launch_to_ready_seconds'],
                      warmup_saved_request_seconds=[x['saved_request_e2e_seconds'] for x in rows if x['role']=='warmup'])
comparisons={}
for group in ['final','cleanup']:
    cells={}
    for prompt in ['short','long']:
        av=[];bv=[];ratios=[]
        for a,b in [('A1','B1'),('A2','B2')]:
            left=visits[f'nomas-49-mean-{group}12-{a}']['per_prompt'][prompt]['raw_seconds']
            right=visits[f'nomas-49-mean-{group}12-{b}']['per_prompt'][prompt]['raw_seconds']
            ratios.extend(x/y for x,y in zip(left,right));av.extend(left);bv.extend(right)
        speedup=statistics.geometric_mean(ratios)
        observed=max(noise['conservative_noise_bound_pct'],summarize(av)['full_range_pct'],summarize(bv)['full_range_pct'])
        cells[prompt]=dict(A=summarize(av),B=summarize(bv),paired_ratios=ratios,
                           paired_geometric_speedup=speedup,latency_reduction_pct=100*(1-1/speedup),
                           observed_descriptive_noise_bound_pct=observed,
                           gain_exceeds_1_5pct_and_noise=100*(1-1/speedup)>max(1.5,observed),
                           cleanup_change_within_noise=abs(100*(1-1/speedup))<=observed,
                           independent_process_visits_per_arm=2)
    comparisons[group]=cells
stage={}
for arm in ['A1','B1']:
    visit=visits[f'nomas-49-mean-stage12-{arm}'];assert visit['sync_stage_profiling']==1
    stage[arm]={name:summarize([x[name] for x in visit['stage_diagnostics']]) for name in ['NomasDenoisingStage','NomasDecodingStage']}
report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),physical_gpus=[0],
            all80_outputs_exact=True,noise10=noise,comparisons=comparisons,visits=visits,
            synchronized_stage_diagnostics=stage,cross_framework_performance_admitted=False,
            cross_framework_quality_evidence='evidence/quality/nomas-49-native02/summary.json',
            BCG=False,capture_cost='not applicable; BCG disabled; intrinsic FlexAttention compilation and first-use verification disclosed in warmups',
            limitations='Same fixed physical GPU0 and identical native BF16 inputs/weights. Two process clusters per final/cleanup arm; all raw samples retained. P95 and observed ranges are descriptive, not confidence bounds. Sync-stage timing is a separate diagnostic phase. No cross-framework speedup admitted; native49 quality failed all6 cells. Memory is native allocator and separately sampled total device, never conflated.')
out=root/'evidence/performance/nomas-mean-final12.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(comparisons=comparisons,stage=stage),indent=2))
