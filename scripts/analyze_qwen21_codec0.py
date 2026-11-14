"""Audit the final head with equal actual PNG encoding; retain every sample."""
import ast
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root / 'evidence/benchmarks/qwen21-codec0-final-01'
assert not out.exists()
receipts = {}
# Preserve the descriptive statistics used before the codec correction.
source = root / 'scripts/analyze_qwen21_prefix_final.py'
names = {'read', 'describe', 'compare', 'aggregate'}
definitions = [n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in definitions} == names
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'))
receipts[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()

controller = read(root / 'evidence/preflight/qwen21-codec0-status.json')
assert controller['status'] == 'codec0_all_measurements_complete_analysis_pending'
rule_path = root / 'inputs/qwen21-codec0-quiet-rule-01.json'
rule = read(rule_path)
assert controller['rule_sha256'] == receipts[str(rule_path.relative_to(root))]
assert rule['actual_png_compress_level'] == 0 and rule['physical_gpus'] == [0]
encoding = read(root / 'evidence/preflight/qwen21-codec0-actual-encoding-equivalence.json')
assert len(encoding) == 18 and all(e['all_deflate_blocks_stored'] for e in encoding)
assert all(e['decoded_RGBA_exact'] for e in encoding if e['mode'] != 'omni')
quality = read(root / 'evidence/quality/qwen21-codec0-main-PR-01/summary.json')
cross_quality = read(root / 'evidence/quality/qwen21-codec0-cross-framework-01/cross-image-edit/metrics.json')
assert quality['numeric_passed'] and cross_quality['numeric_passed']
earlier = read(root / 'evidence/benchmarks/qwen21-prefix-final-01/summary.json')
equivalence = read(root / 'evidence/preflight/qwen21-prefix-format-only-equivalence.json')
assert equivalence['all_runtime_files_git_identical'] and equivalence['test_AST_exact']
assert equivalence['release_head'] == rule['heads']['PR']

visits = []
for visit in rule['visits']:
    folder = root / 'runs' / visit['run']
    status = read(folder / 'status.json')
    plan = read(folder / 'visit.json')
    rows = read(folder / 'outputs/results.json')
    assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
    assert status['benchmark'] and not status['profiler']
    assert status['head_sha'] == rule['heads'][visit['mode']]
    assert receipts[str((folder / 'visit.json').relative_to(root))] == rule['plan_sha256'][visit['plan']]
    assert plan['physical_gpus'] == [0] and not plan.get('synchronized_stage_profiling')
    assert plan['actual_png_compress_level'] == 0
    assert plan['output_compression'] == {'sgl': 0, 'omni': 100}
    assert len(rows) == 9 and [r['role'] for r in rows] == ['warmup'] * 2 + ['measure'] * 7
    assert all(r['matches_reference'] is True and r['native_steps_observed'] == 50 for r in rows)
    metrics = []
    for row in rows:
        directory = root / Path(row['output']).parent.relative_to('/task')
        metrics.append(read(directory / 'perf.json')['memory_checkpoints'] if visit['arm'] == 'sgl'
                       else read(directory / 'response.json').get('metrics'))
    path = folder / 'gpu-samples.jsonl'
    data = path.read_bytes()
    receipts[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    samples = [json.loads(line) for line in data.splitlines()]
    span = (rows[-1]['timer_ns']['materialized'] - rows[2]['timer_ns']['start']) / 1e9
    visits.append(dict(**visit, head_sha=status['head_sha'],
        measured_serial_client_wall_seconds=span,
        measured_seconds=[r['saved_request_e2e_seconds'] for r in rows[2:]],
        warmup_seconds=[r['saved_request_e2e_seconds'] for r in rows[:2]],
        launch_to_ready_seconds=status['launch_to_ready_seconds'],
        sampled_device_peak_MiB=max(float(s['gpu'].splitlines()[0].split(', ')[2]) for s in samples),
        native_memory_or_response_metrics=metrics))
assert len(visits) == 10
aa = {mode: aggregate([v for v in visits if v['phase'] == 'AA' and v['mode'] == mode])
      for mode in ('eager', 'main')}
arms = {mode: aggregate([v for v in visits if v['phase'] == 'paired' and v['mode'] == mode])
        for mode in ('omni', 'main', 'PR')}
comparisons = {name: compare(arms[a]['raw_seconds'], arms[b]['raw_seconds']) for name, a, b in (
    ('main_over_PR', 'main', 'PR'), ('omni_over_PR', 'omni', 'PR'), ('omni_over_main', 'omni', 'main'))}
noise = max(aa['main']['observed_noise_floor_fraction'],
            arms['main']['observed_noise_floor_fraction'], arms['PR']['observed_noise_floor_fraction'])
# Omni's existing A/A used actual level0 already; keep its earlier time window explicit.
cross_noise = max(noise, earlier['omni_AA']['observed_noise_floor_fraction'],
                  arms['omni']['observed_noise_floor_fraction'])
memory = {}
for mode in ('eager', 'main', 'PR', 'omni'):
    selected = [v for v in visits if v['mode'] == mode and
                v['phase'] == ('AA' if mode == 'eager' else 'paired')]
    metrics = [m for v in selected for m in v['native_memory_or_response_metrics'][2:]]
    values = dict(sampled_device_peak_MiB=max(v['sampled_device_peak_MiB'] for v in selected))
    if mode == 'omni':
        values['native_peak_RESERVED_MiB'] = max(m['peak_memory_mb'] for m in metrics)
    else:
        for key in ('peak_allocated_mb', 'peak_reserved_mb'):
            values[key.replace('_mb', '_MiB')] = max(m['runtime_peak'][key] for m in metrics)
    memory[mode] = values

admissions = read(root / 'evidence/preflight/qwen21-prefix-release-admissions-validated.json')
assert admissions['head_sha'] == rule['heads']['PR']
assert all(v['capture_fusion_calls'] == [32, 32, 32] for v in admissions['high_capture_intervals'])
bcg = {}
for mode, run in [('main', 'qwen21-codec0-main-admission-01'), ('PR', 'qwen21-codec0-PR-admission-01')]:
    folder = root / 'runs' / run
    status = read(folder / 'status.json')
    assert status['head_sha'] == rule['heads'][mode] and status['gpu_cleanup'] == 'idle_verified'
    events = []
    for path in (folder / 'bcg-audit').glob('*.jsonl'):
        data = path.read_bytes()
        receipts[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        events.extend(json.loads(line) for line in data.splitlines())
    starts = {(e['pid'], e['call_id'], e['runner'], e['signature']): e['wall_ns']
              for e in events if e['event'] == 'capture_enter' and not e.get('already_present')}
    captures = [e for e in events if e['event'] == 'capture_return' and e['added']]
    serving = [e for e in events if e.get('context', {}).get('is_warmup') is False]
    replays = [e for e in serving if e['event'] == 'graph_replay_return']
    assert len(captures) == 3 and len(replays) == 294
    assert not any(e['event'] == 'capture_enter' or e.get('leaf_mismatch_fallback') for e in serving)
    bcg[mode] = dict(actual_capture_count=3, actual_serving_replays=294, serving_captures=0,
        launch_to_ready_seconds_with_audit=status['launch_to_ready_seconds'],
        capture_wall_seconds_with_audit=[(e['wall_ns'] - starts[e['pid'], e['call_id'], e['runner'], e['signature']]) / 1e9 for e in captures])

stage_visits = []
for mode, index in [('main', 1), ('PR', 1), ('PR', 2), ('main', 2)]:
    folder = root / 'runs' / f'qwen21-codec0-{mode}-stages-{index:02d}'
    status = read(folder / 'status.json')
    plan = read(folder / 'visit.json')
    rows = read(folder / 'outputs/results.json')
    assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
    assert status['head_sha'] == rule['heads'][mode] and not status['profiler']
    assert plan['synchronized_stage_profiling'] and len(rows) == 5
    entries = []
    for row in rows:
        assert row['matches_reference'] and row['native_steps_observed'] == 50
        perf = read(root / Path(row['output']).parent.relative_to('/task') / 'perf.json')
        assert len(perf['denoise_steps_ms']) == 50
        stages = {s['name']: s['duration_ms'] / 1000 for s in perf['steps']}
        entries.append(dict(index=row['index'], role=row['role'], stages_seconds=stages,
                            instrumented_e2e_seconds=row['saved_request_e2e_seconds'],
                            denoise_seconds=stages['QwenImage21DenoisingStage']))
    stage_visits.append(dict(mode=mode, run=folder.name, rows=entries))
stage_summary = {mode: describe([r['denoise_seconds'] for v in stage_visits if v['mode'] == mode
                                for r in v['rows'] if r['role'] == 'measure']) for mode in ('main', 'PR')}
profiles = {}
for mode in ('main', 'PR'):
    run = f'qwen21-codec0-{mode}-profile-01'
    status = read(root / 'runs' / run / 'status.json')
    assert status['head_sha'] == rule['heads'][mode] and status['status'] == 'raw_complete'
    assert status['profiler'] and not status['benchmark']
    profiles[mode] = {prompt: read(root / 'evidence/profiles' / run / prompt / 'raw-costs.json')
                      for prompt in ('image-long', 'image-edit')}
    assert all(not q['benchmark'] for q in profiles[mode].values())

report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), rule=rule,
    visits=visits, AA=aa, paired=arms, comparisons=comparisons,
    observed_main_PR_noise_floor_fraction=noise, observed_cross_framework_noise_floor_fraction=cross_noise,
    final_gain_exceeds_observed_floor=comparisons['main_over_PR']['latency_reduction_fraction'] > noise,
    cross_framework_gain_exceeds_observed_floor=comparisons['omni_over_PR']['latency_reduction_fraction'] > cross_noise,
    codec_equivalence=encoding, quality=quality, cross_quality=cross_quality,
    cleanup_comparison_prior_separate_phase=earlier['comparisons']['prototype_over_cleanup'],
    cleanup_within_prior_observed_floor=earlier['cleanup_change_within_observed_floor'],
    source_equivalence=equivalence, original_Omni_AA_level0=earlier['omni_AA'],
    memory=memory, bcg=bcg, high_capture_interval_audit=admissions,
    synchronized_stage_visits=stage_visits, synchronized_denoise=stage_summary,
    profiles=profiles, all90_quiet_outputs_exact_to_admitted_arm=True,
    all20_stage_outputs_exact_to_admitted_arm=True, receipts=receipts,
    qualified_PR=False, new_beneficial_BCG_scenario=False,
    cross_framework_performance_eligible=True,
    limitations=[
        'Only the frozen 1024x1024, 50-step, BF16, batch1/concurrency1 edit cell passes the cross-framework quality gate. Three T2I failures remain visible and are ineligible for cross-framework speed claims.',
        'Two fresh processes and 14 measured requests per arm, with two warmups per process. O-M-F-F-M-O is balanced in time; all samples are retained. Noise floors and p95 are descriptive, not confidence intervals.',
        'Omni original A/A already used actual PNG level0; it is retained from an earlier time window, not pooled with the final paired samples.',
        'Both use actual PNG level0 with exact unchanged decoded RGBA. SGLang API compression0 and Omni API compression100 map to equivalent encoding. Older 100/100 cross-framework E2E is excluded; same-framework cleanup comparisons remain valid.',
        'Eager AA provides separate context, not a paired final-PR/eager benefit proof. No new beneficial BCG scenario is counted.',
        'Saved-request E2E includes synchronous POST, raw JSON write, base64 decode and PNG write without fsync; pixel validation and hashing are excluded. Serial-client throughput includes validation gaps; inverse mean latency is separate.',
        'Native cuDNN attention is explicit for SGLang; native Omni regional compile and VAE graphs remain enabled. SGLang has no torch.compile or offload. No globally fastest configuration claim.',
        'Synchronized stage timings and Torch profiles are separate from quiet performance. Denoise includes the native SGLang loop and synchronization; Omni default stage/steps is not used as denoise.',
        'NVML is sampled total-device memory; Omni peak_memory_mb means reserved MiB. Do not compare it to SGLang allocated memory.',
        'Launch-to-ready is fresh-process startup with reusable on-disk kernel and OS weight caches. Capture wall durations include audit instrumentation and are diagnostic.',
        'Cleanup comparison4b45/718 is a separate balanced codec9 phase. Final15b differs from718 only in test formatting with identical runtime files and test AST, then repeats all model admissions and the full final codec0 performance/profile run.',
        'Batch2 and uncaptured shape eager fallbacks, CFG2 and request lifecycle were separately validated. No claim for Cache-DiT, quantized graphs, multi-rank graphs or concurrent requests.',
    ])
out.mkdir(parents=True)
(out / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('AA', 'paired', 'comparisons', 'memory', 'synchronized_denoise',
    'final_gain_exceeds_observed_floor', 'cross_framework_gain_exceeds_observed_floor')}, indent=2))
