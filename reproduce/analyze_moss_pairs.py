"""Analyze the predeclared symmetric three-arm MOSS serving protocol."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values, p):
    values=sorted(values); pos=(len(values)-1)*p; lo=int(pos); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('plan',type=Path)
    parser.add_argument('--evidence',type=Path,default=Path('evidence'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text()); blocks=[]; by_arm={}
    expected=['main_cudnn','candidate_cudnn','omni','omni','candidate_cudnn','main_cudnn']*2
    assert plan['order']==expected
    missing=[]
    for i, config in enumerate(plan['order']):
        name=f'{plan["id"]}-{i:02d}-{config}'
        paths=[args.evidence/'benchmarks'/name/'summary.json',args.evidence/'block-runs'/name/'metadata.json']
        missing += [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit('Incomplete protocol; no aggregate gain claim:\n'+'\n'.join(missing))
    for i, config in enumerate(plan['order']):
        name=f'{plan["id"]}-{i:02d}-{config}'
        folder=args.evidence/'benchmarks'/name
        meta=json.loads((folder/'metadata.json').read_text())
        server=json.loads((args.evidence/'block-runs'/name/'metadata.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        records=json.loads((folder/'results.json').read_text())
        measured=[r for r in records if not r['warmup']]
        assert len(measured)==plan['repeats'] and len(records)==plan['warmups']+plan['repeats']
        assert not meta['profiler'] and meta['source_diff_sha256']=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        expected_sha = (plan['candidate_head_sha'] if config.startswith('candidate') else
                        '817d4c17feb702289a2d2b2cbacfd594c262b7a7' if config == 'omni' else
                        'af21e2b989e482cfdce39689126562f2209d53ac')
        assert meta['source_sha']==expected_sha
        assert not (folder/'source-status.txt').read_text().strip()
        assert meta['input_sha256']=='5025043d914c5299bb21083ca0cf559c39eb1ce91a458fbc687017a9aec29d0c'
        assert summary['all_outputs_byte_identical']
        arm='omni' if config=='omni' else 'sgl'
        quality=json.loads((args.evidence/'qualification'/f'{arm}-moss-{name}-quality.json').read_text())
        assert quality['all_pass']
        qualified=json.loads((args.evidence/'qualification'/f'{arm}-moss-{name}'/'results.json').read_text())
        assert all(r['sha256']==qualified[0]['sha256'] for r in records)
        assert qualified[0]['sha256']==qualified[-1]['sha256']
        values=[r['http_seconds'] for r in measured]
        samples=[]
        allowed=set(meta['args']['allowed_gpu_pids'])
        for row in records:
            assert hashlib.sha256((folder/f'{row["index"]:02d}'/'output.png').read_bytes()).hexdigest()==row['sha256']
            gpu_rows=json.loads((folder/f'{row["index"]:02d}'/'gpu-samples.json').read_text())
            for sample in gpu_rows:
                first,second=sample['gpus']
                assert first['uuid']=='GPU-1565b45e-a702-2015-9776-61f68c36fd59'
                assert second['uuid']=='GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd'
                assert set(first['pids']) <= allowed and not second['pids']
            if not row['warmup']:
                samples.extend(s['gpus'][0] for s in gpu_rows)
        stage=[]; denoise=[]; queue=[]
        for row in measured:
            request_folder=folder/f'{row["index"]:02d}'
            if arm=='sgl':
                perf=json.loads((request_folder/'perf.json').read_text())
                stage.append(perf['total_duration_ms']/1000)
                denoise.append(next(s['duration_ms']/1000 for s in perf['steps'] if s['name']=='MOSSImageDenoisingStage'))
            else:
                response=json.loads((request_folder/'response.json').read_text())
                durations=response['metrics']['stage_durations']
                stage.append(durations['stage_0_gen_ms']/1000)
                queue.append(durations['queue_wait_ms']/1000)
        block=dict(index=i,config=config,name=name,p50_seconds=statistics.median(values),http_seconds=values,
                   denoise_seconds=denoise,engine_total_seconds=stage,queue_seconds=queue,
                   peak_gpu_memory_bytes=max(r['peak_gpu_memory_bytes'] for r in measured),
                   launch_to_ready_seconds=server['launch_to_ready_seconds'],quality_pass=True,
                   source_sha=meta['source_sha'],output_sha256=records[0]['sha256'],
                   warmup_http_seconds=[r['http_seconds'] for r in records if r['warmup']],
                   measured_cv_percent=100*statistics.stdev(values)/statistics.mean(values),
                   temperature_c_range=[min(s['temperature_c'] for s in samples),max(s['temperature_c'] for s in samples)],
                   sm_clock_mhz_range=[min(s['sm_clock_mhz'] for s in samples),max(s['sm_clock_mhz'] for s in samples)],
                   mean_sampled_power_w=statistics.mean(s['power_mw']/1000 for s in samples),
                   ownership_check='All archived warmup/measured NVML samples: expected GPU UUID0, only recorded owned PIDs; GPU1 no processes.')
        blocks.append(block); by_arm.setdefault(config,[]).append(block)
    aggregates={}
    for config,group in by_arm.items():
        values=[v for b in group for v in b['http_seconds']]
        denoise=[v for b in group for v in b['denoise_seconds']]
        engine=[v for b in group for v in b['engine_total_seconds']]
        aggregates[config]=dict(process_count=len(group),measured_request_count=len(values),
            http_p50_seconds=statistics.median(values),http_p95_seconds=quantile(values,.95),
            sequential_requests_per_second=len(values)/sum(values),
            denoise_p50_seconds=statistics.median(denoise) if denoise else None,
            denoise_definition='SGL native denoising stage includes scheduler; Omni denoise requires separate native instrumentation',
            engine_total_p50_seconds=statistics.median(engine),
            peak_gpu_memory_bytes=max(b['peak_gpu_memory_bytes'] for b in group),
            process_launch_to_ready_seconds=[b['launch_to_ready_seconds'] for b in group],
            process_output_sha256=[b['output_sha256'] for b in group],
            thermal_warmup_total_http_seconds=[sum(b['warmup_http_seconds']) for b in group],
            measured_block_cv_percent=[b['measured_cv_percent'] for b in group],
            capture_cost_seconds=None,
            capture_note='SGL BCG disabled. Omni regional compile is native; process launch-to-ready plus request warmup recorded separately, no isolated cold compile estimate.')
    pair_sets={
        'main_over_candidate':[(0,1),(5,4),(6,7),(11,10)],
        'omni_over_candidate':[(2,1),(3,4),(8,7),(9,10)],
        'omni_over_main':[(2,0),(3,5),(8,6),(9,11)],
    }
    paired={}
    for label,index_pairs in pair_sets.items():
        pairs=[dict(numerator_block=i,denominator_block=j,
                    speedup=blocks[i]['p50_seconds']/blocks[j]['p50_seconds']) for i,j in index_pairs]
        ratios=[p['speedup'] for p in pairs];rng=random.Random(1101)
        bootstrap=[geometric_mean(rng.choices(ratios,k=len(ratios))) for _ in range(10000)]
        paired[label]=dict(pairs=pairs,geometric_mean_speedup=geometric_mean(ratios),
            all_pairs_faster=all(r>1 for r in ratios),min_pair_speedup=min(ratios),max_pair_speedup=max(ratios),
            descriptive_pair_bootstrap_95_interval=[quantile(bootstrap,.025),quantile(bootstrap,.975)],
            interval_caveat='Only4 temporal process pairs; exploratory resampling of pairs, not a claim of broad workload uncertainty.')
    output=dict(protocol=plan,blocks=blocks,aggregates=aggregates,paired=paired,
        scope='Only frozen MOSS Base1024²/50/CFG3 batch1 concurrency1 hot conditioning. Not the full81entrymatrix; no BCG claim.',
        omitted_from_end_to_end_timing='Server startup/thermal warmups and client PNG decode/write are excluded; HTTP fullresponse bytes included.',
        note='Every planned block retained. Profiler-free timing. Native Omni denoise diagnostic and final media/review remain separate gates.')
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(aggregates=aggregates,paired=paired),indent=2))


if __name__=='__main__':
    main()
