"""Independently inspect every archived request, ownership sample and original statistic."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'evidence'
RUN = 'sana-video-depthwise-final-primary-cleanup1'
ARCHIVE = 'sana-video-depthwise-final-primary1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def percentile(values,fraction):
    ordered = sorted(values)
    position = (len(values)-1)*fraction
    low = int(position)
    return ordered[low]+(ordered[min(low+1,len(values)-1)]-ordered[low])*(position-low)


plan_path = E/'plans'/(RUN+'.json')
plan = read(plan_path)
assert digest(plan_path) == '8149e5fbb6c94117df941bc5f7502bed974dd0997be5889950bc2bf38c2131d4'
assert plan['SGL_SYNC_STAGE_PROFILING'] == 0 and plan['quality'] == 'lossless'
archive_path = E/'checkpoints'/(ARCHIVE+'-local-verification.json')
verified = read(archive_path)
assert verified['all_member_hashes_verified'] and verified['all_members_extracted']
reproduction = E/'reproduction'/ARCHIVE
assert digest(reproduction/(RUN+'.json')) == digest(plan_path)
status = read(reproduction/(RUN+'-status.json'))
assert status['status'] == 'raw_complete_audit_pending' and len(status['completed']) == len(plan['order']) == 12
for name,expected in plan['sha256'].items():
    source = reproduction/Path(name).name if name.startswith('scripts/') else ROOT/name
    assert digest(source) == expected,name
for name in plan['required_admissions']:
    assert read(ROOT/name)['passed'],name
admission_path = E/'preflight'/(RUN+'-admission.json')
admission = read(admission_path)
assert admission['plan_sha256'] == digest(plan_path)
assert admission['script_sha256'] == plan['sha256']['scripts/admit_sana_video_depthwise_final_primary_cleanup1.py']
contract = read(ROOT/plan['contract'])
native_quality = read(E/'qualification/sgl-sana-video-t2v-native1-vs-omni-sana-video-t2v-native1-v2-quality.json')
uuids = ['GPU-1565b45e-a702-2015-9776-61f68c36fd59','GPU-7ca33735-71c8-fc89-57a6-c7cdbf8d28cd']
requests, blocks = [], []
for index,config in enumerate(plan['order']):
    name = f'{RUN}-{index:02d}-{config}'
    spec = plan['configurations'][config]
    arm = spec['arm']
    head = spec.get('head','af21e2b989e482cfdce39689126562f2209d53ac' if arm == 'sgl' else '817d4c17feb702289a2d2b2cbacfd594c262b7a7')
    patch_sha = spec.get('diff_sha256',hashlib.sha256(b'').hexdigest())
    block = E/'block-runs'/name
    meta = read(block/'metadata.json')
    assert meta['plan'] == plan and meta['index'] == index and meta['config'] == config and meta['status'] == 'complete'
    assert meta['source_sha'] == head and meta['source_diff_sha256'] == patch_sha == digest(block/'source.patch')
    assert meta['runner_sha256'] == plan['sha256']['scripts/run_sana_video_blocks1.py']
    assert digest(block/'launch.sh') == plan['sha256']['scripts/'+spec['launcher']]
    if arm == 'sgl':
        assert 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0' in (block/'launch.sh').read_text()
        assert not any(marker in (block/'server.log').read_text(errors='replace') for marker in ['Falling back to diffusers backend','Using diffusers backend','Loaded diffusers pipeline'])
    reference = E/'qualification'/spec['reference']
    assert digest(reference/'results.json') == spec['reference_results_sha256']
    reference_rows = read(reference/'results.json')
    prompt_order = plan['prompt_orders'].get(str(index),plan['measured_prompts'])
    assert [command[command.index('--prompt-id')+1] for command in meta['commands']] == prompt_order
    for prompt in plan['measured_prompts']:
        quality = [row for row in native_quality['rows'] if row['request'].endswith('-'+prompt)]
        assert quality and all(row['all_frames_pass'] for row in quality)
        references = [row for row in reference_rows if Path(row['request']).name.endswith('-'+prompt)]
        assert references and all(row['sha256'] == spec['output_sha256'][prompt] for row in references)
        for row in references:
            assert digest(reference/Path(row['request']).name/'output.mp4') == row['sha256']
            assert row['frames'] == len(row['frame_sha256']) == 81
        if arm == 'sgl':
            assert spec['output_sha256'][prompt] == plan['configurations']['main_eager']['output_sha256'][prompt]
        folder = E/'benchmarks'/(name+'-'+prompt)
        m,rows,summary = (read(folder/(key+'.json')) for key in ('metadata','results','summary'))
        assert m['source_sha'] == head and m['source_repo'] == spec['repo'] and m['source_diff_sha256'] == patch_sha
        if patch_sha != hashlib.sha256(b'').hexdigest():
            assert digest(folder/'source.patch') == patch_sha
        assert m['harness_sha256'] == plan['sha256']['scripts/bench_sana_video_serving1.py']
        assert m['input_sha256'] == digest(ROOT/plan['contract']) and not m['profiler']
        assert m['physical_gpus'] == [0] and m['physical_gpu_pool'] == [0,1]
        assert m['driver'] == '595.84' and m['torch'] == ('2.14.1+cu130' if arm == 'sgl' else '2.13.0+cu130')
        options = m['args']
        assert options['warmups'] == plan['warmups'] and options['repeats'] == plan['repeats']
        assert options['quality'] == 'lossless' and options['prompt_id'] == prompt and not options['diagnostic_profiler']
        assert options['expected_output_sha256'] == spec['output_sha256'][prompt]
        allowed = set(options['allowed_gpu_pids'])
        assert allowed and allowed == set(meta['gpu_container_pids'][0]) and not meta['gpu_container_pids'][1]
        assert {int(pid) for pid in m['allowed_process_commands']} == allowed
        assert all('vllm' in value or 'sgl_diffusion' in value for value in m['allowed_process_commands'].values())
        assert len(rows) == plan['warmups']+plan['repeats'] == 4
        assert [row['index'] for row in rows] == list(range(4))
        values,peaks = [],[]
        for row in rows:
            request_folder = folder/f'{row["index"]:02d}'
            request = read(request_folder/'request.json')
            for key in ['width','height','num_frames','fps','num_inference_steps','guidance_scale','seed','negative_prompt']:
                assert request[key] == contract[key],key
            text = next(item['prompt'] for item in contract['prompts'] if item['id'] == prompt)
            assert request['model'] == contract['model_path']
            assert request['prompt'] == text+(' motion score: 30.' if arm == 'sgl' else '')
            assert row['warmup'] == (row['index'] < plan['warmups'])
            if arm == 'sgl':
                expected = dict(generator_device='cpu',max_sequence_length=300,quality='lossless',output_compression=100,x264_preset='ultrafast',enhance_prompt=False,profile=False)
                assert all(request[key] == value for key,value in expected.items())
                perf = read(request_folder/'perf.json')
                terminal = read(request_folder/'terminal-response.json')
                assert perf['commit_hash'] == head and len(perf['denoise_steps_ms']) == 50
                assert terminal['id'] == perf['request_id'] and terminal['status'] == 'completed' and terminal['num_outputs'] == 1 and terminal['error'] is None
                assert terminal['size'] == '832x480' and row['poll_count'] > 0
            else:
                assert json.loads(request['extra_params']) == contract['encoding']['omni_extra_params']
                assert row['poll_count'] == 0
            video = request_folder/'output.mp4'
            assert digest(video) == row['sha256'] == spec['output_sha256'][prompt]
            headers = {key.lower():value for key,value in read(request_folder/'response-headers.json').items()}
            assert 'video/mp4' in headers['content-type']
            if 'content-length' in headers:
                assert int(headers['content-length']) == video.stat().st_size
            samples = read(request_folder/'gpu-samples.json')
            assert len(samples) >= 2 and all(a['monotonic'] < b['monotonic'] for a,b in zip(samples,samples[1:]))
            assert samples[-1]['monotonic']-samples[0]['monotonic'] >= row['http_seconds']
            for sample in [m['initial_gpu_state'],*samples]:
                g0,g1 = sample['gpus']
                assert [g0['uuid'],g1['uuid']] == uuids and [g0['index'],g1['index']] == [0,1]
                assert set(g0['pids']) <= allowed and not g1['pids']
            peak = max(sample['gpus'][0]['memory_bytes'] for sample in samples)
            assert peak == row['peak_gpu_memory_bytes']
            requests.append(dict(block=index,configuration=config,prompt=prompt,index=row['index'],warmup=row['warmup'],
                http_seconds=row['http_seconds'],MP4_sha256=row['sha256'],gpu_samples=len(samples),observed_peak_gpu_memory_bytes=peak))
            if not row['warmup']:
                values.append(row['http_seconds']);peaks.append(peak)
        assert len(values) == summary['n'] == 3
        assert summary['mean_seconds'] == statistics.mean(values) and summary['p50_seconds'] == statistics.median(values)
        assert summary['p95_seconds'] == percentile(values,.95) and summary['peak_gpu_memory_bytes'] == max(peaks)
        blocks.append(dict(index=index,configuration=config,prompt=prompt,raw_seconds=values,mean_seconds=statistics.mean(values),
            median_seconds=statistics.median(values),peak_gpu_memory_bytes=max(peaks),launch_to_ready_seconds=meta['launch_to_ready_seconds']))
assert blocks == admission['blocks'] and len(requests) == 96
for result in admission['rows']:
    selected = [row for row in blocks if row['prompt'] == result['prompt']]
    by_index = {row['index']:row for row in selected}
    for config,n in plan['n_measured_per_arm_per_prompt'].items():
        values = [value for row in selected if row['configuration'] == config for value in row['raw_seconds']]
        arm = result['arms'][config]
        assert len(values) == n == arm['n'] and statistics.mean(values) == arm['mean_seconds']
        assert statistics.median(values) == arm['p50_seconds'] and percentile(values,.95) == arm['p95_seconds']
    for key,indices in plan['pair_indices'].items():
        ratios = [by_index[a]['median_seconds']/by_index[b]['median_seconds'] for a,b in indices]
        pair = result['paired'][key]
        assert pair['pairs'] == indices and pair['process_median_ratios'] == ratios
        logs = [math.log(value) for value in ratios]
        center = statistics.mean(logs)
        radius = {2:12.706204736432095,4:3.182446305284263}[len(logs)]*statistics.stdev(logs)/math.sqrt(len(logs))
        assert pair['geometric_mean_ratio'] == math.exp(center)
        assert pair['paired_log_t95_interval'] == [math.exp(center-radius),math.exp(center+radius)]
    assert result['paired']['omni_vs_final']['all_pairs_faster'], 'No stable native Omni advantage'
out = E/'preflight/sana-video-depthwise-final-primary1-raw-audit.json'
assert not out.exists()
result = dict(passed=True,raw_request_count=96,measured_request_count=sum(not row['warmup'] for row in requests),
    all_original_source_requests_outputs_and_gpu_samples_audited=True,all_original_statistics_recomputed=True,
    physical_gpu_UUIDs=uuids,all_samples_retained=True,performance_admission_passed=admission['passed'],
    source_sha='f248594caca5c533c94612b6848ff3512ed98e08',plan_sha256=digest(plan_path),admission_sha256=digest(admission_path),
    archive_verification_sha256=digest(archive_path),script_sha256=digest(Path(__file__)),requests=requests,
    limitations='Offline audit after collection; no changes to frozen order, samples, thresholds, model or timing. Raw source/request/telemetry correctness is separate from the original performance admission. MP4 equality propagates previously verified native frame identity; these96videos were not independently redecoded or visually reviewed in full.')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key != 'requests'},indent=2))
