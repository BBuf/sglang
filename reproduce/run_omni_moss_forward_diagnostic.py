"""Measure actual MOSS full-forward methods using native profiler targets only."""
import json

import run_moss_final_diagnostics as harness

harness.RUN = 'moss-qkv-final-omni-forward-profile1'
PATCH = '77dd31e4ec32c402d1ad5912c6e17fc0947ddcfb198b4c42564d4812e38bebac'
REPO = '/task/diagnostics/omni-moss-forward-profile'
harness.assert_protocol_complete()
with harness.server('omni-forward', 'omni', REPO, harness.OMNI,
                    'launch_omni_moss_forward_diagnostic.sh', PATCH) as (run, pids):
    expected_image = harness.qualify(run, 'omni', harness.RUN, 'high', [
        'sgl-moss-moss-qkv-final-paired1-00-main_cudnn',
        'sgl-moss-moss-qkv-final-paired1-01-candidate_cudnn',
    ])
    run('native-forward-stage-aa', ['python', '/task/bench_moss_serving.py', 'omni',
        '--run-id', harness.RUN, '--repo', REPO, '--expected-sha', harness.OMNI,
        '--expected-diff-sha256', PATCH, '--allowed-gpu-pids', *map(str, pids),
        '--quality', 'high', '--configuration-label', 'diagnostic-only-native-forward-profiler-targets',
        '--warmups', '4', '--repeats', '4'])
    rows = json.loads((harness.ROOT / 'outputs/benchmarks' / harness.RUN / 'results.json').read_text())
    assert all(row['sha256'] == expected_image for row in rows)
    for row in rows:
        response = json.loads((harness.ROOT / 'outputs/benchmarks' / harness.RUN / f'{row["index"]:02d}' / 'response.json').read_text())
        stages = response['metrics']['stage_durations']
        assert stages['MOSSImagePipeline.predict_noise_with_slg'] > 0
        assert stages['MOSSImagePipeline.scheduler_step_maybe_with_cfg'] > 0
print(json.dumps(dict(status='native_forward_denoise_timing_completed', primary_benchmark=False)), flush=True)
