"""Read final-commit mode switching, matched traces and separate native Omni timing."""
import hashlib
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
RUN = 'moss-qkv-final-diagnostics1'
OMNI_RUN = 'moss-qkv-final-omni-forward-profile1'


def read(path):
    return json.loads(path.read_text())


def main():
    quality = {}
    for label in ('main-lossless', 'main-high', 'candidate-lossless', 'candidate-high', 'candidate-lossless-after-high'):
        folder = EVIDENCE / 'qualification' / f'sgl-moss-{RUN}-{label}'
        gate = read(folder.with_name(folder.name + '-quality.json'))
        rows = read(folder / 'results.json')
        assert gate['all_pass'] and len(rows) == 4
        assert rows[0]['sha256'] == rows[-1]['sha256']
        for row in rows:
            prompt = Path(row['request']).name
            assert hashlib.sha256((folder / prompt / 'output.png').read_bytes()).hexdigest() == row['sha256']
        if label != 'main-lossless':
            assert all(r['pixel_identical'] for r in gate['results'])
        quality[label] = dict(report=str(folder.with_name(folder.name + '-quality.json').relative_to(ROOT)),
                              output_sha256=[r['sha256'] for r in rows],
                              all_reference_pixels_identical=all(r['pixel_identical'] for r in gate['results']))

    profiles = {}
    for arm in ('main', 'candidate'):
        traces = list((EVIDENCE / 'profiles' / f'{RUN}-{arm}').glob('*.trace.json.gz'))
        assert len(traces) == 1, traces
        target = EVIDENCE / 'profiles' / f'{RUN}-{arm}-summary.json'
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_moss_trace.py'), str(traces[0]), '--output', str(target)], check=True)
        report = read(target)
        qualified = quality[arm + '-high']['output_sha256'][0]
        generated = read(EVIDENCE / 'qualification' / f'sgl-moss-{RUN}-{arm}-profile' / 'results.json')
        assert generated[0]['sha256'] == qualified
        selected = [s for s in report['steps'] if s['name'] in ('ProfilerStep#2', 'ProfilerStep#3', 'ProfilerStep#4')]
        assert len(selected) == 3
        groups = {
            'gemm': lambda n: 'gemm' in n.lower(),
            'bf16_copy': lambda n: 'copy' in n.lower() and 'bfloat16' in n.lower(),
            'norm_rope': lambda n: '_norm_rope_kernel' in n,
            'cudnn_attention': lambda n: 'cudnn_generated' in n and 'sdpa' in n,
        }
        rows = []
        for step in selected:
            row = dict(step=step['name'], gpu_annotation_ms=step['gpu_annotation_ms'])
            for name, match in groups.items():
                kernels = [k for k in step['kernels'] if match(k['name'])]
                row[name] = dict(calls=sum(k['calls'] for k in kernels), kernel_sum_ms=sum(k['sum_ms'] for k in kernels))
            rows.append(row)
        profiles[arm] = dict(trace=str(traces[0].relative_to(ROOT)), trace_sha256=report['trace_sha256'],
                             cuda_graph_launch_count=report['cuda_graph_launch_count'],
                             profile_png_matches_qualified_high=True, matched_complete_gpu_steps=rows)

    omni = EVIDENCE / 'benchmarks' / OMNI_RUN
    records = read(omni / 'results.json')
    assert len(records) == 8 and len({r['sha256'] for r in records}) == 1
    rows = []
    for record in records:
        if record['warmup']:
            continue
        response = read(omni / f'{record["index"]:02d}' / 'response.json')
        durations = response['metrics']['stage_durations']
        predict = durations['MOSSImagePipeline.predict_noise_with_slg']
        scheduler = durations['MOSSImagePipeline.scheduler_step_maybe_with_cfg']
        value = predict + scheduler
        assert 0 < value < record['http_seconds']
        rows.append(dict(index=record['index'], denoise_core_seconds=value,
                         predict_seconds=predict, scheduler_seconds=scheduler,
                         http_seconds=record['http_seconds'], all_native_stage_durations=durations))
    assert len(rows) == 4
    result = dict(final_head='d752066eaa6f5091fa4c8daa2ba1d5d43a3928cd', quality=quality, profiles=profiles,
                  omni_native_stage=dict(rows=rows, denoise_core_p50_seconds=statistics.median(r['denoise_core_seconds'] for r in rows),
                      http_p50_seconds=statistics.median(r['http_seconds'] for r in rows),
                      definition='Sum of native profiler durations for actual full-forward predict_noise_with_slg and scheduler_step_maybe_with_cfg calls over50 steps. Includes prediction+Euler update, excludes surrounding Python-loop/kwargs/setup overhead. Seconds, unlike stage_0_gen_ms. An isolated diagnostic checkout only adds these existing profiler targets; primary source/timing unchanged.',
                      instrumentation_patch_sha256='77dd31e4ec32c402d1ad5912c6e17fc0947ddcfb198b4c42564d4812e38bebac',
                      prior_diagnostic_limitation='Default native profiler wraps denoise_step/diffuse, but actual MOSS full-forward bypasses those methods; diagnostics2 only returned encoder/VAE timing. Retained, not used as denoise measurement.',
                      primary_latency_comparison=False),
                  profile_caveat='Only complete GPU annotations #2/#3/#4 compared; profiler timings are diagnostic and never replace profiler-free HTTP latency. Native Omni stage timing is separate and has a different boundary from SGL denoise stage.')
    path = EVIDENCE / 'benchmarks' / 'moss-qkv-final-diagnostics-summary.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=RUN)
    parser.add_argument('--omni-run-id', default=OMNI_RUN)
    args = parser.parse_args()
    RUN, OMNI_RUN = args.run_id, args.omni_run_id
    main()
