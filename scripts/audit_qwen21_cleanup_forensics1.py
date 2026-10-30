"""Verify every forensic PNG/request and separate synchronized stage evidence."""
import base64
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'evidence'
RUN = 'qwen21-cleanup-1024-forensics1'
EMPTY = hashlib.sha256(b'').hexdigest()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan = read(E/'plans'/(RUN+'.json'))
verification_path = E/'checkpoints'/(RUN+'-local-verification.json')
verification = read(verification_path)
assert verification['all_member_hashes_verified'] and verification['all_members_extracted']
assert read(ROOT/plan['preserved_failure']['admission'])['passed'] is False
for path, expected in plan['sha256'].items():
    assert digest(ROOT/path) == expected, path
for path in plan['required_admissions']:
    assert read(ROOT/path)['passed'], path
assert not plan['primary_timing'] and plan['diagnostic_timing']
assert plan['order'] == ['original', 'current'] and plan['expected_outputs'] == 44
images, tasks = [], {}


def request_audit(payload, contract, prompt, profiled):
    for key in ['seed', 'num_inference_steps', 'true_cfg_scale', 'negative_prompt', 'generator_device']:
        assert str(payload[key]) == str(contract[key]), key
    assert payload['size'] == '1024x1024' and payload['quality'] == 'high'
    assert payload['prompt'] == prompt and payload['model'] == contract['model_path']
    assert int(payload['n']) == 1 and int(payload['output_compression']) == 0
    assert payload['response_format'] == 'b64_json' and payload['output_format'] == 'png'
    assert str(payload.get('profile', False)).lower() == str(profiled).lower()
    if profiled:
        assert int(payload['num_profiled_timesteps']) == 3


def image_audit(directory, row, expected_sha):
    png = (directory/'output.png').read_bytes()
    assert hashlib.sha256(png).hexdigest() == row['sha256'] == expected_sha
    response = read(directory/'response.json')
    assert len(response['data']) == 1
    assert base64.b64decode(response['data'][0]['b64_json'], validate=True) == png
    with Image.open(directory/'output.png') as im:
        assert im.mode == 'RGBA' and im.size == (1024, 1024)
    images.append(dict(path=str((directory/'output.png').relative_to(ROOT)), sha256=expected_sha))


for index, config in enumerate(plan['order']):
    spec = plan['configurations'][config]
    name = f'{RUN}-{index:02d}-{config}'
    server = E/'block-runs'/name
    meta = read(server/'metadata.json')
    assert meta['plan'] == plan and meta['source_sha'] == spec['head'] and meta['status'] == 'complete'
    assert meta['source_diff_sha256'] == EMPTY and not (server/'source.patch').read_bytes()
    assert 'SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1' in (server/'launch.sh').read_text()
    log = (server/'server.log').read_text(errors='replace')
    assert not any(s in log for s in ['Falling back to diffusers backend', 'Using diffusers backend', 'Loaded diffusers pipeline'])
    assert meta['gpu_container_pids'][0] and not meta['gpu_container_pids'][1]
    tasks[config] = {}
    for task in plan['tasks']:
        contract = read(ROOT/'inputs'/plan['contracts'][task])
        reference = E/'qualification'/plan['main_qualification_references'][task]
        refs = read(reference/'results.json')
        assert len(refs) == 4 and refs[0]['sha256'] == refs[3]['sha256']
        for profiled in [False, True]:
            suffix = 'profile-high' if profiled else 'high'
            folder = E/'qualification'/f'sgl-qwen21-base-{task}-{name}-{suffix}'
            rows, qm = read(folder/'results.json'), read(folder/'metadata.json')
            assert len(rows) == (1 if profiled else 4)
            assert qm['source_sha'] == spec['head'] and qm['source_diff_sha256'] == EMPTY
            assert qm['input_sha256'] == plan['contract_sha256'][task] and qm['profiled'] == profiled
            assert qm['benchmark'] is False
            assert digest(folder/'contract.json') == plan['contract_sha256'][task]
            for i, row in enumerate(rows):
                directory = folder/Path(row['request']).name
                request = read(directory/'request.json')
                request_audit(request['payload'], contract, contract['prompts'][i % 3]['prompt'], profiled)
                assert request['endpoint'] == ('edits' if task == 'edit' else 'generations')
                assert request['reference_image'] == contract.get('reference_image')
                image_audit(directory, row, refs[i]['sha256'])
                assert row['profiled'] == profiled and row['benchmark'] is False
                perf = read(directory/'perf.json')
                assert perf['commit_hash'] == spec['head'] and len(perf['denoise_steps_ms']) == 50
                if profiled:
                    traces = list((E/'profiles'/name).glob(perf['request_id']+'*.trace.json.gz'))
                    assert len(traces) == 1
                    tasks[config][task] = dict(profile_trace=str(traces[0].relative_to(ROOT)),
                        profile_sha256=digest(traces[0]), profile_request=str(directory.relative_to(ROOT)),
                        profile_output_sha256=row['sha256'])
        folder = E/'benchmarks'/f'{name}-{task}'
        bm, rows = read(folder/'metadata.json'), read(folder/'results.json')
        assert bm['source_sha'] == spec['head'] and bm['source_diff_sha256'] == EMPTY
        assert bm['profiler'] and bm['input_sha256'] == plan['contract_sha256'][task]
        assert len(rows) == plan['warmups'] + plan['repeats'] == 6
        allowed = set(bm['args']['allowed_gpu_pids'])
        assert allowed == set(meta['gpu_container_pids'][0])
        denoise, memory = [], []
        for i, row in enumerate(rows):
            directory = folder/f'{i:02d}'
            request_audit(read(directory/'request.json'), contract, contract['prompts'][0]['prompt'], False)
            image_audit(directory, row, refs[0]['sha256'])
            assert row['warmup'] == (i < plan['warmups'])
            for sample in read(directory/'gpu-samples.json'):
                assert sample['gpus'][0]['uuid'] == plan['physical_gpu_uuid']
                assert set(sample['gpus'][0]['pids']) <= allowed and not sample['gpus'][1]['pids']
                memory.append(sample['gpus'][0]['memory_bytes'])
            perf = read(directory/'perf.json')
            assert perf['commit_hash'] == spec['head'] and len(perf['denoise_steps_ms']) == 50
            durations = [x['duration_ms']/1000 for x in perf['steps'] if x['name'] == 'QwenImage21DenoisingStage']
            assert len(durations) == 1 and 0 < durations[0] < row['http_seconds']
            if not row['warmup']:
                denoise.append(durations[0])
        tasks[config][task].update(denoise_seconds=denoise, denoise_p50_seconds=statistics.median(denoise),
                                  peak_sampled_memory_bytes=max(memory), excluded_warmups=plan['warmups'])
assert len(images) == len({x['path'] for x in images}) == plan['expected_outputs']
out = E/'preflight'/(RUN+'-audit.json')
assert not out.exists()
out.write_text(json.dumps(dict(all_raw_audits_pass=True, primary_timing=False, failed_primary_remains_failed=True,
    archive_verification_sha256=digest(verification_path), source_heads={c:s['head'] for c,s in plan['configurations'].items()},
    images=images, tasks=tasks,
    caveat='One fresh original/current process, synchronized stage diagnostics and separate Torch profile requests. Exact 44 native RGBA images checked against same-request primary main. No primary timing admission, retry or new optimization claim.'), indent=2)+'\n')
print(json.dumps(dict(images=len(images), tasks=tasks, failed_primary_remains_failed=True)))
