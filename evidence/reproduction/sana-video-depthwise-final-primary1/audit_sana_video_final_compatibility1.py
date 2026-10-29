"""Audit final native videos and observed graph lifecycle, without timing claims."""
import hashlib
import json
from pathlib import Path

root = Path('/task')
run = 'sana-video-ln-final-compatibility1'
plan_path = root/(run+'-plan-v2.json')
plan = json.loads(plan_path.read_text())
status = json.loads((root/(run+'-status.json')).read_text())
assert status['status']=='raw_complete_quality_replay_and_profile_audit_pending'
assert len(status['completed'])==4
sequence = json.loads((root/'sana-video-final-quality-sequence1.json').read_text())
contract = json.loads((root/'inputs/sana-video-480p-t2v-workloads.json').read_text())
original = json.loads((root/'outputs/qualification/sgl-sana-video-t2v-native1/results.json').read_text())
original_by_prompt = {Path(row['request']).name.split('-',1)[1]:row for row in original}
main_by_condition = {}
rows = []
for index, configuration in enumerate(plan['order']):
    name = f'{run}-{index:02d}-{configuration}'
    spec = plan['configurations'][configuration]
    folder = root/'block-runs'/name
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['status']=='complete_raw_audit_pending'
    assert meta['source_sha']==spec['head'] and meta['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
    files = list((root/'outputs/diagnostics'/name).glob('sgl-worker-*.jsonl'))
    assert len(files)==1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert all(row['hook_sha256']==plan['sha256']['scripts/sana_video_bcg_sitecustomize1.py'] for row in records)
    warmups = [row for row in records if row['is_warmup']]
    serving = [row for row in records if not row['is_warmup']]
    expected_requests = 12+int(configuration in plan['profile_configurations'])
    assert len(serving)==expected_requests
    serving_by_id = {row['request_id']:row for row in serving}
    assert len(serving_by_id)==len(serving)
    assert all(row['capture_calls']==0 and not row['actual_captures'] for row in serving)
    captures = [capture for row in warmups for capture in row['actual_captures']]
    if spec['BCG']:
        assert captures and all(capture['segments']>0 and capture['wall_seconds']>0 for capture in captures)
    else:
        assert not captures
    outputs = []
    for profiled in ([False,True] if configuration in plan['profile_configurations'] else [False]):
        label = name+('-profile' if profiled else '')
        directory = root/'outputs/qualification'/('sgl-sana-video-'+label)
        results = json.loads((directory/'results.json').read_text())
        declared = [dict(prompt_id='short',quality='lossless')] if profiled else sequence
        assert len(results)==len(declared)
        for result, request in zip(results, declared):
            assert result['prompt_id']==request['prompt_id'] and result['quality']==request['quality']
            assert result['profiled']==profiled and result['frames']==81
            output = directory/Path(result['request']).name/'output.mp4'
            assert hashlib.sha256(output.read_bytes()).hexdigest()==result['sha256']
            perf = json.loads((output.parent/'perf.json').read_text())
            assert len(perf['denoise_steps_ms'])==50
            observed = serving_by_id[perf['request_id']]
            assert observed['quality']==request['quality'] and observed['mounted_quality']==request['quality']
            assert (observed['width'],observed['height'],observed['num_frames'])==(832,480,81)
            assert observed['guidance_scale']==6
            assert not observed['ln_gate']['disabled'] and not observed['ln_gate']['wrapper_disabled']
            expected_replays = 100 if spec['BCG'] and request['quality']=='lossless' else 0
            assert observed['replay_calls']==expected_replays
            runners = {row['quality']:row['entries'] for row in observed['runners']}
            if spec['BCG']:
                assert runners['lossless']>0 and all(value==0 for key,value in runners.items() if key!='lossless')
                assert observed['ln_gate']['signatures']
            elif configuration=='final_eager':
                assert observed['ln_gate']['signatures']
            key = (request['quality'],request['prompt_id'])
            if configuration=='main_eager':
                if key in main_by_condition:
                    assert result['sha256']==main_by_condition[key]['sha256']
                else:
                    main_by_condition[key] = result
            reference = main_by_condition[key]
            assert result['sha256']==reference['sha256']
            assert result['frame_sha256']==reference['frame_sha256']
            if request['quality']=='lossless':
                native = original_by_prompt[request['prompt_id']]
                assert result['sha256']==native['sha256'] and result['frame_sha256']==native['frame_sha256']
            outputs.append(dict(path=str(output),quality=request['quality'],prompt=request['prompt_id'],sha256=result['sha256'],
                                request_id=observed['request_id'],profiled=profiled,replay_calls=observed['replay_calls'],
                                segments_per_replay=observed['segments_per_replay'],ln_gate=observed['ln_gate'],same_quality_main_MP4_and_all81frames_exact=True))
    rows.append(dict(configuration=configuration,source_sha=spec['head'],observer_sha256=meta['observer_sha256'],
                     launch_to_ready_seconds=meta['launch_to_ready_seconds'],warmup_requests=len(warmups),
                     actual_warmup_captures=captures,warmup_capture_method_calls=sum(row['capture_calls'] for row in warmups),
                     warmup_denoising_wall_seconds=[(row['finished_ns']-row['started_ns'])/1e9 for row in warmups],
                     serving_requests=len(serving),serving_capture_calls=0,serving_replay_calls=sum(row['replay_calls'] for row in serving),
                     gpu_samples=str(folder/'gpu-samples.csv.json'),outputs=outputs))
assert sum(len(row['outputs']) for row in rows)==51
result = dict(passed=True,rows=rows,benchmark=False,new_BCG_support_claim=False,final_primary_admitted=False,
              remaining='Manually attribute actual final eager and main/final BCG profile kernel paths, inspect real final media, then produce finalcompatibilityadmission. No primary timing based on these instrumented runs.',
              scope='832x480,81frames16fps,50steps,CFG6,CPUseed1101,300tokens,batch/concurrency1,GPU0;12qualitysequence perarm plus3profiles. Exact/high BCG requests intentionally use uncaptured quality-specific eager runners; returnlossless reuses captured graph. Chinese crossOmni PSNRfailure retained.',
              plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = root/'evidence/preflight/sana-video-ln-final-compatibility1-raw-audit.json'
assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='rows'},indent=2))
