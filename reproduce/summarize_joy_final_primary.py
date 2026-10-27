"""Aggregate fully audited final-head primary blocks without pooling diagnostics."""
import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_joy_blocks import percentile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shape', choices=['square','wide'])
    args = parser.parse_args()
    run = f'joy-gelu-final-{args.shape}-paired1'
    analysis_path = EVIDENCE/'benchmarks'/(run+'-analysis.json')
    analysis = read(analysis_path)
    assert analysis['all_planned_blocks_present'] and analysis['source_and_ownership_audit_pass']
    assert analysis['plan']['candidate_head_sha'] == '4c135ea92d6b4cbb0ead69bb39976f7dbe995455'
    assert analysis['plan']['primary_performance_evidence']
    assert not analysis['plan'].get('diagnostic_profiler')
    blocks = analysis['blocks']
    assert [b['config'] for b in blocks] == ['main','candidate','omni_compile','omni_compile','candidate','main']
    aa_path = EVIDENCE/'benchmarks'/('joy-gelu-v1-aa1-analysis.json' if args.shape=='square'
                                   else 'joy-gelu-final-wide-aa1-analysis.json')
    aa = read(aa_path)
    assert aa['aa']['passed'] and aa['source_and_ownership_audit_pass']
    noise = max(aa['aa']['block_median_relative_difference'], aa['aa']['max_within_block_cv_percent']/100)
    arms = {}
    for arm in ('main','candidate','omni_compile'):
        selected = [b for b in blocks if b['config']==arm]
        assert len(selected)==2 and all(b['ownership_and_output_audit_pass'] for b in selected)
        values = [x for b in selected for x in b['http_seconds']]
        denoise = [x for b in selected for x in b['denoise_seconds']]
        assert len(values)==8
        assert len(denoise)==(0 if arm=='omni_compile' else 8)
        arms[arm] = dict(process_count=2, measured_request_count=len(values),
            http_p50_seconds=statistics.median(values), http_p95_seconds=percentile(values,.95),
            sequential_requests_per_second=len(values)/sum(values),
            denoise_p50_seconds=statistics.median(denoise) if denoise else None,
            denoise_scope='SGL native synchronized DenoisingStage; Omni separately instrumented diagnostics pending',
            peak_gpu_memory_bytes=max(b['peak_gpu_memory_bytes'] for b in selected),
            block_median_seconds=[b['p50_seconds'] for b in selected],
            max_within_block_cv_percent=max(b['cv_percent'] for b in selected),
            startup_and_warmup=[dict(block=b['name'], launch_to_ready_seconds=b['launch_to_ready_seconds'],
                four_quality_request_seconds=b['prebenchmark_quality_http_seconds'],
                two_explicit_warmup_seconds=b['warmup_http_seconds']) for b in selected])
    pairs = {}
    for label, item in analysis['paired'].items():
        ratios = [blocks[i]['p50_seconds']/blocks[j]['p50_seconds'] for i,j in item['pair_indices']]
        assert ratios==item['ratios']
        geo = math.exp(statistics.mean(math.log(x) for x in ratios))
        assert geo==item['geometric_mean']
        pairs[label] = dict(item, latency_reduction_percent=100*(1-1/geo),
                           all_pairs_exceed_observed_aa_envelope=all(x-1>noise for x in ratios))
    main_medians = arms['main']['block_median_seconds']
    endpoint_difference = max(main_medians)/min(main_medians)-1
    endpoint_check = analysis['plan']['main_endpoint_noise_check']
    result = dict(shape=args.shape, primary_only=True, source_and_output_audit_pass=True,
        analysis=str(analysis_path.relative_to(ROOT)), plan=analysis['plan'], scope=analysis['scope'],
        arms=arms, paired=pairs, aa_analysis=str(aa_path.relative_to(ROOT)),
        observed_aa_envelope_fraction=noise, main_endpoint_relative_difference=endpoint_difference,
        main_endpoint_check=endpoint_check,
        main_endpoint_noise_passed=endpoint_difference <= endpoint_check['max_relative_difference'],
        final_diagnostics_and_precision_audit_pending=True, official_pr_qualified=False,
        caveats=['Two fresh processes/eight measured requests per arm; p95 is a descriptive sample percentile, not a production SLO.',
                 'Sequential req/s is reciprocal mean service latency at concurrency1, not saturated throughput.',
                 'Startup uses existing disk/compiler caches. Four quality requests precede two explicit warmups in every process.',
                 'All planned blocks retained; no profiler data in HTTP primary aggregates.',
                 'The A/A envelope comparison is a descriptive noise screen, not a confidence interval or significance test.',
                 'This cell does not establish complete frozen-matrix coverage or a BCG performance win.'])
    destination = EVIDENCE/'benchmarks'/(run+'-summary.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(shape=args.shape,paired=pairs,main_endpoint_relative_difference=endpoint_difference,
                         arms={k:{name:v[name] for name in ('http_p50_seconds','http_p95_seconds','denoise_p50_seconds',
                              'sequential_requests_per_second','peak_gpu_memory_bytes')} for k,v in arms.items()}),indent=2))


if __name__ == '__main__':
    main()
