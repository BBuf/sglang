"""Admit archived current-head measurements after the complete raw Joy audit."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('shape', choices=['square', 'wide'])
parser.add_argument('phase', choices=['aa', 'primary'])
args = parser.parse_args()
run = f'joy-current-head-{args.shape}-{args.phase}1'
plan_path = root/'evidence/plans'/(run+'.json')
analysis_path = root/'evidence/benchmarks'/(run+'-analysis.json')
plan = json.loads(plan_path.read_text())
analysis = json.loads(analysis_path.read_text())
assert analysis['plan'] == plan
assert analysis['source_and_ownership_audit_pass'] and analysis['all_planned_blocks_present']
blocks = analysis['blocks']
assert len(blocks) == len(plan['order'])
assert all(block['ownership_and_output_audit_pass'] for block in blocks)
for name in plan['required_admissions']:
    assert json.loads((root/name).read_text())['passed'], name
verified_path = root/'evidence/checkpoints'/(run+'-local-verification.json')
verified = json.loads(verified_path.read_text())
assert verified['all_member_hashes_verified'] and verified['all_members_extracted']
result = dict(run=run, source_sha=plan['current_head'], phase=args.phase,
              all_raw_images_and_ownership_audited=True, final_PR_admitted=False,
              primary_stage_times_valid=False,
              stage_timing_note='SYNC0 primary perf dumps are retained as raw data; stage latency must come from separate synchronized diagnostics.',
              plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
              archive_verification_sha256=hashlib.sha256(verified_path.read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if args.phase == 'aa':
    assert len(blocks) == 2 and all(row['config'] == 'main' for row in blocks)
    noise = 100*analysis['aa']['block_median_relative_difference']
    result.update(passed=analysis['aa']['passed'], max_noise_percent=noise,
                  max_within_process_cv_percent=analysis['aa']['max_within_block_cv_percent'])
else:
    aa_path = root/'evidence/preflight'/f'joy-current-head-{args.shape}-aa1-admission.json'
    aa = json.loads(aa_path.read_text())
    assert aa['passed'] and len(blocks) == 12
    means, arms, pairs = {}, {}, {}
    for arm in plan['configurations']:
        selected = [row for row in blocks if row['config'] == arm]
        values = [value for row in selected for value in row['http_seconds']]
        assert len(values) == plan['n_measured_per_arm'][arm]
        means[arm] = statistics.mean(values)
        ordered = sorted(values)
        position = (len(values)-1)*.95
        lo = int(position)
        p95 = ordered[lo]+(ordered[min(lo+1, len(values)-1)]-ordered[lo])*(position-lo)
        arms[arm] = dict(n=len(values), mean_seconds=means[arm],
                         p50_seconds=statistics.median(values), p95_seconds=p95,
                         serial_response_rate_per_second=1/means[arm],
                         peak_gpu_memory_bytes=max(row['peak_gpu_memory_bytes'] for row in selected),
                         process_medians=[row['p50_seconds'] for row in selected])
    for name, indices in plan['pair_indices'].items():
        ratios = [blocks[left]['p50_seconds']/blocks[right]['p50_seconds'] for left, right in indices]
        logs = [math.log(value) for value in ratios]
        t = {2: 12.706204736432095, 4: 3.182446305284263}[len(logs)]
        center = statistics.mean(logs)
        radius = t*statistics.stdev(logs)/math.sqrt(len(logs))
        pairs[name] = dict(indices=indices, ratios=ratios, geometric_mean_ratio=math.exp(center),
                           log_t95_interval=[math.exp(center-radius), math.exp(center+radius)],
                           all_pairs_faster=all(value > 1 for value in ratios), df=len(logs)-1)
    gain = 100*(1-means['candidate_current']/means['main'])
    cleanup = 100*(means['candidate_current']/means['candidate_original']-1)
    cleanup_upper = 100*(1/pairs['original_over_current']['log_t95_interval'][0]-1)
    primary_pass = (gain >= plan['admission']['minimum_main_latency_reduction_percent']
                    and gain > aa['max_noise_percent']
                    and pairs['main_over_current']['all_pairs_faster']
                    and pairs['omni_over_current']['all_pairs_faster'])
    cleanup_pass = (cleanup <= aa['max_noise_percent']
                    and cleanup_upper < plan['admission']['cleanup_t95_regression_upper_percent_max'])
    result.update(passed=primary_pass and cleanup_pass, primary_passed=primary_pass,
                  cleanup_passed=cleanup_pass, AAmax_noise_percent=aa['max_noise_percent'],
                  arms=arms, pairs=pairs, final_vs_main_latency_reduction_percent=gain,
                  omni_over_current_speedup=means['omni_compile']/means['candidate_current'],
                  cleanup_mean_latency_change_percent=cleanup,
                  cleanup_t95_regression_upper_percent=cleanup_upper,
                  note='All predeclared samples and process pairs retained. A failed/uncertain gate needs explicit supplemental evidence; no sample dropping or threshold changes. This does not admit missing diagnostics, compatibility, media or PR publication.')
out = root/'evidence/preflight'/(run+'-admission.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: value for key, value in result.items() if key not in ['arms', 'pairs']}, indent=2))
