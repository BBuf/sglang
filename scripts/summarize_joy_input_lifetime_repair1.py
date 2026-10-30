"""Summarize admitted isolated repair evidence without mixing historical timings."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
head = '03368cf1c325e526184b8fb223443e1ae931b13c'
sources = []


def read(name):
    path = evidence/name
    sources.append(path)
    return json.loads(path.read_text())


tests = read('preflight/joy-input-lifetime-repair1-tests-admission.json')
diagnostics = read('preflight/joy-input-lifetime-repair1-diagnostics1-admission.json')
assert tests['passed'] and tests['tests_passed'] == 1 and tests['tests_skipped'] == 0
assert diagnostics['passed'] and diagnostics['all_raw_outputs_rehashed'] == 76
assert tests['source_sha'] == diagnostics['source_sha'] == head
assert tests['red_before_fix'] and tests['green_after_fix']
rows, comparisons = [], []
lines = [
    'Candidate current in these tables is the isolated repair03368cf1c325e526184b8fb223443e1ae931b13c, not official merged PR47022 headb08c24e9630c63790c9f86b48d6c7fc10a0c4301. The official-head square cleanup failure remains recorded; no qualification count or new optimization slot is assigned.', '',
    'All primary values below are new SYNC0 requests. Historical Joy timings used SYNC1 and are not pooled with these measurements.', '',
    '| Shape | Configuration | n | Mean / p50 / p95 HTTP E2E (s) | Serial req/s | Observed request peak GiB |',
    '|---|---|---:|---:|---:|---:|',
]
for shape in ('square', 'wide'):
    aa = read(f'preflight/joy-input-lifetime-repair1-{shape}-aa1-admission.json')
    primary = read(f'preflight/joy-input-lifetime-repair1-{shape}-primary1-admission.json')
    assert aa['passed'] and primary['passed'] and primary['source_sha'] == head
    for phase in ('aa', 'primary'):
        verification = read(f'checkpoints/joy-input-lifetime-repair1-{shape}-{phase}1-local-verification.json')
        assert verification['all_member_hashes_verified'] and verification['all_members_extracted']
    for arm in ('main', 'candidate_original', 'candidate_current', 'omni_compile'):
        value = primary['arms'][arm]
        lines.append(f'| {shape} | {arm} | {value["n"]} | {value["mean_seconds"]:.4f} / {value["p50_seconds"]:.4f} / {value["p95_seconds"]:.4f} | {value["serial_response_rate_per_second"]:.5f} | {value["peak_gpu_memory_bytes"]/1024**3:.3f} |')
        rows.append(dict(shape=shape, configuration=arm, **value))
    comparisons.append(dict(shape=shape, **{key:primary[key] for key in (
        'AAmax_noise_percent', 'final_vs_main_latency_reduction_percent', 'omni_over_current_speedup',
        'cleanup_mean_latency_change_percent', 'cleanup_t95_regression_upper_percent', 'pairs')}))
lines += ['', '| Shape | A/A noise | Repair E2E reduction vs main | Omni / repair | Original → repair mean latency change | Cleanup paired-t95 upper regression |',
          '|---|---:|---:|---:|---:|---:|']
for row in comparisons:
    lines.append(f'| {row["shape"]} | {row["AAmax_noise_percent"]:.6f}% | {row["final_vs_main_latency_reduction_percent"]:.6f}% | {row["omni_over_current_speedup"]:.6f}× | {row["cleanup_mean_latency_change_percent"]:+.6f}% | {row["cleanup_t95_regression_upper_percent"]:.6f}% |')
lines += ['', '| Shape | Comparison | Process pairs | Paired median geometric ratio | Log-t95 interval |',
          '|---|---|---:|---:|---:|']
for row in comparisons:
    for name, pair in row['pairs'].items():
        low, high = pair['log_t95_interval']
        lines.append(f'| {row["shape"]} | {name} | {len(pair["indices"])} | {pair["geometric_mean_ratio"]:.6f} | [{low:.6f}, {high:.6f}] |')
lines += ['', 'Every frozen request and process pair is retained. Main and native Omni have two process replicates per shape; original/current have four. Intervals use process pairs, not individual requests as independent replicates. NVML peaks are sampled observations, not allocator maxima. Serial response rate is 1/mean E2E at concurrency1.', '',
          '| Shape | Source | Separate synchronized short denoise p50 (s) | Stage samples | Cached launch-to-ready (s) |',
          '|---|---|---:|---:|---:|']
for row in diagnostics['synchronized_diagnostic_stages']:
    lines.append(f'| {row["shape"]} | {row["source"]} | {row["denoise_p50_seconds"]:.4f} | {row["n_short"]} | {row["launch_to_ready_seconds"]:.3f} |')
lines += ['', 'These SYNC1 stage values come from separate lossless→high→lossless diagnostic processes and never replace primary E2E. Tall is a quality/shape check only. Previously published native Omni stage diagnostics remain historical; no new Omni stage measurement is claimed.', '',
          '| Profile | Kernels | Kernel sum (ms) | ATen GELU calls | Fused custom-op calls | Fused activation kernels |',
          '|---|---:|---:|---:|---:|---:|']
for name, row in diagnostics['profiles'].items():
    lines.append(f'| {name} | {row["kernel_count"]} | {row["kernel_sum_ms"]:.3f} | {row["gelu_cpu_calls"]} | {row["fused_custom_op_calls"]} | {row["activation_calls"]} |')
quality = []
lines += ['', '| Shape | Quality | Samples across phases | Pixel-exact samples | Minimum SSIM | Minimum finite PSNR (dB) |',
          '|---|---|---:|---:|---:|---:|']
for shape in ('square', 'wide', 'tall'):
    for mode in ('high', 'lossless'):
        selected = [x for x in diagnostics['quality_rows'] if x['shape'] == shape and x['quality'] == mode]
        assert len(selected) == (4 if mode == 'high' else 8) and all(x['passed'] for x in selected)
        finite = [x['psnr_db'] for x in selected if x['psnr_db'] is not None]
        minimum = min(finite) if finite else None
        row = dict(shape=shape, quality=mode, samples=len(selected), exact=sum(x['pixel_identical'] for x in selected),
                   min_ssim=min(x['ssim'] for x in selected), min_finite_psnr_db=minimum,
                   worst_ssim_request=min(selected, key=lambda x:x['ssim']))
        quality.append(row)
        psnr = f'{minimum:.5f}' if minimum is not None else 'all infinite'
        lines.append(f'| {shape} | {mode} | {row["samples"]} | {row["exact"]} | {row["min_ssim"]:.6f} | {psnr} |')
lines += ['', 'All native outputs reproduce the pinned same-source references. Lossless compares exact RGB pixels; high checks every request against SSIM≥0.95 and PSNR≥28dB. Short/long/Chinese/repeated-short requests and return to lossless are retained. These data do not establish new BCG, GPU compile performance or full-model LoRA support. Actual main/repair media review and a publication decision remain separate gates.']
output = evidence/'joy-input-lifetime-repair1-performance-table1.md'
assert not output.exists()
output.write_text('\n'.join(lines)+'\n')
summary = dict(source_sha=head, previous_tested_head='4c135ea92d6b4cbb0ead69bb39976f7dbe995455',
               main_sha='af21e2b989e482cfdce39689126562f2209d53ac', omni_sha='817d4c17feb702289a2d2b2cbacfd594c262b7a7',
               numerical_admission_passed=True, official_PR_current_head_qualified=False,
               official_PR_head='b08c24e9630c63790c9f86b48d6c7fc10a0c4301', new_optimization_slot_assigned=False, no_historical_primary_pooling=True,
               primary_rows=rows, comparisons=comparisons, quality=quality,
               source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               table_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path = evidence/'preflight/joy-input-lifetime-repair1-summary1.json'
assert not path.exists()
path.write_text(json.dumps(summary, indent=2)+'\n')
print(output.read_text())
