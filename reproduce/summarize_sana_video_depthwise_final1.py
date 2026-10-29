"""Summarize final unprofiled samples without recomputing or relaxing admission."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'evidence'
admission_path = E/'preflight/sana-video-depthwise-final-primary-cleanup1-admission.json'
admission = json.loads(admission_path.read_text())
assert admission['passed'], 'Keep failed original admission; explicitly plan any supplement first'
raw_audit_path = E/'preflight/sana-video-depthwise-final-primary1-raw-audit.json'
raw_audit = json.loads(raw_audit_path.read_text())
assert raw_audit['passed'] and raw_audit['raw_request_count'] == 96
assert raw_audit['admission_sha256'] == hashlib.sha256(admission_path.read_bytes()).hexdigest()
compat_path = E/'preflight/sana-video-depthwise-final-compatibility1-admission.json'
compat = json.loads(compat_path.read_text())
assert compat['passed']
omni_path = E/'preflight/sana-video-omni-stages1-admission.json'
omni = json.loads(omni_path.read_text())
assert omni['passed'] and omni['source_sha']=='817d4c17feb702289a2d2b2cbacfd594c262b7a7'
assert compat['source_sha']=='f248594caca5c533c94612b6848ff3512ed98e08'
costs = dict(rows=list(compat['rows']))
omni_stages = []
for prompt in ['short', 'long']:
    samples = [row for row in omni['rows'] if row['prompt']==prompt and not row['excluded_first_request']]
    assert len(samples)==1 and samples[0]['native_MP4_exact'] and samples[0]['all81frame_hashes_exact']
    stage = samples[0]['stages']
    omni_stages.append(dict(prompt=prompt,quality='lossless',n=1,
                            mean_denoise_seconds=stage['diffuse'],mean_decode_seconds=stage['vae.decode'],
                            scope='Previously archived native Omni stage diagnostic at the unchanged frozen source/checkpoint/configuration; primary E2E below is newly measured.'))
costs['rows'].append(dict(configuration='omni_eager',diagnostic_stages=omni_stages))
rows = []
table = [
    '| Configuration | Prompt | n | Mean / p50 / p95 E2E (s) | Serial req/s | Request peak GiB | Separate synchronized denoise / decode (s) |',
    '|---|---|---:|---:|---:|---:|---:|',
]
for arm in ['omni_eager','main_eager','v1_eager','final_eager']:
    for result in admission['rows']:
        prompt = result['prompt']
        values = result['arms'][arm]
        diagnostic = [stage for row in costs['rows'] if row['configuration']==arm
                      for stage in row['diagnostic_stages'] if stage['prompt']==prompt and stage['quality']=='lossless']
        assert len(diagnostic)<=1
        denoise = diagnostic[0]['mean_denoise_seconds'] if diagnostic else None
        peaks_gib = values['peak_gpu_memory_bytes']/1024**3
        stage_text = f'{denoise:.4f} / {diagnostic[0]["mean_decode_seconds"]:.4f} (n={diagnostic[0]["n"]})' if denoise is not None else 'not collected here'
        table.append(f'| {arm} | {prompt} | {values["n"]} | {values["mean_seconds"]:.4f} / {values["p50_seconds"]:.4f} / {values["p95_seconds"]:.4f} | {values["serial_response_rate_per_second"]:.5f} | {peaks_gib:.3f} | {stage_text} |')
        rows.append(dict(configuration=arm,prompt=prompt,primary=values,separate_synchronized_denoise=diagnostic[0] if diagnostic else None))
table.extend(['', 'Primary values use every predeclared measured request. NVML is sampled every 250 ms during requests; these are observed device-memory peaks, not exact allocator maxima. Denoise is from a separate synchronized diagnostic, not a component subtracted from primary E2E. V1 is retained for cleanup comparison. Omni denoise/decode reuses the already archived native stage diagnostic at the unchanged frozen source/checkpoint/configuration, with one steady request per English prompt. All primary E2E samples are new. Native stage boundaries differ across frameworks; no diagnostic timing replaces primary E2E.', '',
    '| Prompt | Final mean E2E reduction vs main | Omni / final mean E2E | V1 → final mean latency change | Cleanup paired-t 95% regression upper bound |',
    '|---|---:|---:|---:|---:|'])
for result in admission['rows']:
    table.append(f'| {result["prompt"]} | {result["final_vs_main_latency_reduction_percent"]:.5f}% | {result["omni_over_final_speedup"]:.6f}× | {result["cleanup_mean_latency_change_percent"]:+.5f}% | {result["cleanup_paired_t95_latency_regression_upper_percent"]:.5f}% |')
table.extend(['', '| Comparison | Prompt | Process pairs | Paired median geometric ratio | Paired-log-t 95% interval |', '|---|---|---:|---:|---:|'])
for result in admission['rows']:
    for name, pair in result['paired'].items():
        low, high = pair['paired_log_t95_interval']
        table.append(f'| {name} | {result["prompt"]} | {len(pair["pairs"])} | {pair["geometric_mean_ratio"]:.6f} | [{low:.6f}, {high:.6f}] |')
table.extend(['', 'Intervals use process-pair log ratios, not requests as independent process replicates; only two main/final and Omni/final process pairs limit precision. Cleanup has four predeclared pairs. All original requests and warmups remain in raw evidence.', '', '| Configuration | Cached launch-to-ready min / max (s) |', '|---|---:|'])
launch = {}
for row in admission['blocks']:
    launch.setdefault(row['configuration'],{})[row['index']] = row['launch_to_ready_seconds']
for arm, samples in launch.items():
    table.append(f'| {arm} | {min(samples.values()):.3f} / {max(samples.values()):.3f} |')
table.extend(['', 'These are cached process launches, not cold-start distributions. BCG capture/replay costs and quality transitions are reported separately; this depthwise fusion does not establish new BCG support or a BCG net win.'])
out = E/'sana-video-depthwise-final-performance-table1.md'
assert not out.exists()
out.write_text('\n'.join(table)+'\n')
sources = [admission_path,compat_path,omni_path,raw_audit_path]
summary = dict(source_sha=compat['source_sha'],main_sha='af21e2b989e482cfdce39689126562f2209d53ac',
    omni_sha='817d4c17feb702289a2d2b2cbacfd594c262b7a7',final_primary_cleanup_passed=True,raw_primary_audit_passed=True,
    new_BCG_support_or_net_win=False,official_PR_qualified=False,rows=rows,
    comparisons=[{key:value for key,value in result.items() if key!='arms'} for result in admission['rows']],
    cached_launch_to_ready_seconds=launch,primary_admission=str(admission_path.relative_to(ROOT)),
    retained_cross_framework_failure='Chinese PSNR below24dB in frames26–28; excluded primary, retained quality/media/matrix.',
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path = E/'preflight/sana-video-depthwise-final-summary1.json'
assert not path.exists()
path.write_text(json.dumps(summary,indent=2)+'\n')
print(out.read_text())
