"""All final quiet visits; timing, quality receipts and limitations stay separate."""
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root / 'evidence/benchmarks/qwen21-prefix-final-01'
assert not out.exists()
receipts = {}


def read(path):
    data = path.read_bytes()
    receipts[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def describe(values):
    ordered = sorted(values)
    median = statistics.median(values)
    pos = .95 * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return dict(n=len(values), raw_seconds=values, p50_seconds=median,
                p95_seconds=ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo),
                mad_seconds=statistics.median(abs(x - median) for x in values),
                inverse_mean_latency_requests_per_second=len(values) / sum(values))


def compare(a, b):
    ratios = [x / y for x, y in zip(a, b, strict=True)]
    speedup = math.exp(statistics.mean(math.log(x) for x in ratios))
    return dict(paired_speedups=ratios, geometric_mean_speedup=speedup,
                latency_reduction_fraction=1 - 1 / speedup,
                minimum_paired_speedup=min(ratios), maximum_paired_speedup=max(ratios))


def aggregate(selected):
    assert len(selected) == 2
    a, b = [v['measured_seconds'] for v in selected]
    # Conservative descriptive floor; not a statistical confidence interval.
    log_floor = max(abs(math.log(statistics.median(b) / statistics.median(a))),
                    math.log(max(a) / min(a)), math.log(max(b) / min(b)))
    return dict(**describe(a + b), observed_noise_floor_fraction=math.expm1(log_floor),
                serial_client_requests_per_second=14 / sum(v['measured_serial_client_wall_seconds'] for v in selected),
                sampled_device_peak_MiB=max(v['sampled_device_peak_MiB'] for v in selected))


rule_path = root / 'inputs/qwen21-prefix-final-quiet-rule-01.json'
rule = read(rule_path)
controller = read(root / 'evidence/preflight/qwen21-prefix-final-quiet01-status.json')
assert controller['status'] == 'final_quiet_measurements_complete_analysis_pending'
assert controller['rule_sha256'] == receipts[str(rule_path.relative_to(root))]
pilot = read(root / rule['baseline_AA'])
visits = []
for v in rule['visits']:
    folder = root / 'runs' / v['run']
    status = read(folder / 'status.json')
    plan = read(folder / 'visit.json')
    rows = read(folder / 'outputs/results.json')
    assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
    assert status['benchmark'] and not status['profiler']
    assert status['head_sha'] == rule[v['mode'] + '_sha']
    assert receipts[str((folder / 'visit.json').relative_to(root))] == rule['plan_sha256'][v['plan']]
    assert plan['physical_gpus'] == [0] and not plan.get('synchronized_stage_profiling')
    assert len(rows) == 9 and [r['role'] for r in rows] == ['warmup'] * 2 + ['measure'] * 7
    assert all(r['matches_reference'] is True and r['native_steps_observed'] == 50 for r in rows)
    metrics = []
    for row in rows:
        directory = root / Path(row['output']).parent.relative_to('/task')
        if v['arm'] == 'sgl':
            metrics.append(read(directory / 'perf.json')['memory_checkpoints'])
        else:
            # Omni's default denoise_step_latency_ms falls back to full stage / steps.
            # Keep it as raw metadata; it is not a synchronized denoise measurement.
            metrics.append(read(directory / 'response.json').get('metrics'))
    samples_path = folder / 'gpu-samples.jsonl'
    data = samples_path.read_bytes()
    receipts[str(samples_path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    samples = [json.loads(line) for line in data.splitlines()]
    peak = max(float(s['gpu'].splitlines()[0].split(', ')[2]) for s in samples)
    span = (rows[-1]['timer_ns']['materialized'] - rows[2]['timer_ns']['start']) / 1e9
    visits.append(dict(**v, head_sha=status['head_sha'],
        measured_serial_client_wall_seconds=span,
        measured_serial_client_requests_per_second=7 / span,
        launch_to_ready_seconds=status['launch_to_ready_seconds'],
        warmup_seconds=[r['saved_request_e2e_seconds'] for r in rows[:2]],
        measured_seconds=[r['saved_request_e2e_seconds'] for r in rows[2:]],
        sampled_device_peak_MiB=peak, native_memory_or_response_metrics=metrics))

assert len(visits) == 10
omni_aa = aggregate([v for v in visits if v['phase'] == 'OmniAA'])
arms = {mode: aggregate([v for v in visits if v['phase'] == 'paired' and v['mode'] == mode])
        for mode in ('omni', 'main', 'prototype', 'cleanup')}
comparisons = {name: compare(arms[a]['raw_seconds'], arms[b]['raw_seconds']) for name, a, b in [
    ('main_over_cleanup', 'main', 'cleanup'),
    ('main_over_prototype', 'main', 'prototype'),
    ('prototype_over_cleanup', 'prototype', 'cleanup'),
    ('omni_over_main', 'omni', 'main'),
    ('omni_over_cleanup', 'omni', 'cleanup')]}
noise = max(pilot['AA']['bcg']['observed_noise_floor_fraction'],
            arms['main']['observed_noise_floor_fraction'], arms['cleanup']['observed_noise_floor_fraction'])
cleanup_delta = abs(comparisons['prototype_over_cleanup']['latency_reduction_fraction'])
report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), rule=rule,
    visits=visits, omni_AA=omni_aa, paired=arms, comparisons=comparisons,
    main_and_cleanup_observed_noise_floor_fraction=noise,
    cleanup_absolute_change_fraction=cleanup_delta,
    cleanup_change_within_observed_floor=cleanup_delta <= max(noise, arms['prototype']['observed_noise_floor_fraction']),
    main_to_cleanup_gain_exceeds_observed_floor=comparisons['main_over_cleanup']['latency_reduction_fraction'] > noise,
    all90_outputs_exact_to_admitted_arm=True, receipts=receipts,
    qualified_PR=False, new_beneficial_BCG_scenario=False,
    limitations=[
        'Two starts and 14 measurements per paired arm. Pair ordinal requests within first/second visits; V-M-P-C-C-P-M-V balances temporal placement but does not remove all drift.',
        'A/A noise floors and p95 are descriptive, not confidence intervals. All samples retained.',
        'Default native Omni fast paths retained. Different outputs pass the preregistered edit quality gate; they are not cross-framework bit-identical.',
        'T2I quality failures remain in the frozen matrix and receive no cross-framework performance claim.',
        'Stage synchronization disabled for saved-request E2E. No denoise claim from asynchronous perf dumps or Omni default stage/steps metrics.',
        'Saved-request E2E includes synchronous POST, raw response write, JSON/base64 decode and PNG write without fsync. PIL decode and hashes excluded.',
        'Serial client throughput includes between-request validation gaps within each seven-request visit. Inverse mean latency is a separate quantity.',
        'NVML peak is sampled total-device memory; native memory metrics have different definitions and are retained separately.',
        'BCG versus eager is not paired here. Earlier pilot showed optimized BCG still slower than eager; no newly beneficial BCG claim.'
    ])
out.mkdir(parents=True)
(out / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['omni_AA', 'paired', 'comparisons', 'cleanup_change_within_observed_floor', 'main_to_cleanup_gain_exceeds_observed_floor']}, indent=2))
