"""Admit final timings only after raw, actual-kernel and actual-media reviews."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
head = 'e0d694593a65cb5b463c4011b84ee50abefa5e43'
paths = dict(raw=evidence/'preflight/sana-video-ln-final-compatibility1-raw-audit.json',
             profiles=evidence/'preflight/sana-video-ln-final-profile1.json',
             tests=evidence/'block-runs/sana-video-ln-final-tests2/results.json',
             cpu=evidence/'diagnostics/sana-video-ln-final-cpu-compile1/result.json',
             media=evidence/'media/sana-video-ln-final-e0d69459/manifest.json',
             visual=evidence/'media/sana-video-ln-final-e0d69459/visual-review.json',
             archive=evidence/'checkpoints/sana-video-ln-qualification1-local-verification.json')
data = {name:json.loads(path.read_text()) for name,path in paths.items()}
assert data['raw']['passed'] and data['profiles']['passed']
assert len(data['tests'])==2 and all(row['returncode']==0 and row['source_sha']==head for row in data['tests'])
assert data['cpu']['passed'] and data['cpu']['source_sha']==head
assert data['archive']['all_member_hashes_verified'] and data['archive']['all_members_extracted']
assert data['visual']['passed'] and data['visual']['source_sha']==head
assert len(data['visual']['reviewed_panels'])==3
assert len(data['media']['rows'])==3
for row in data['media']['rows']:
    panel = root/row['panel']
    assert str(panel.relative_to(root)) in data['visual']['reviewed_panels']
    assert hashlib.sha256(panel.read_bytes()).hexdigest()==row['panel_sha256']
    assert row['MP4_exact'] and row['all81frames_exact']
    assert row['inputs'][1]['source_sha']==head

rows = []
for arm in data['raw']['rows']:
    sample_path = evidence/Path(arm['gpu_samples']).relative_to('/task')
    samples_gpu = json.loads(sample_path.read_text())
    memory_mib = []
    for sample in samples_gpu:
        devices = [line.split(',') for line in sample['csv'].strip().splitlines()]
        assert devices[0][0].strip() == 'GPU-1565b45e-a702-2015-9776-61f68c36fd59'
        memory_mib.append(float(devices[0][1].strip().split()[0]))
    assert memory_mib
    groups = {}
    for output in arm['outputs']:
        if output['profiled']:
            continue
        remote_path = Path(output['path']).relative_to('/task/outputs')
        path = evidence/remote_path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==output['sha256']
        perf = json.loads((path.parent/'perf.json').read_text())
        stages = {row['name']:row['duration_ms'] for row in perf['steps']}
        assert len(perf['denoise_steps_ms'])==50
        key = (output['quality'],output['prompt'])
        groups.setdefault(key,[]).append(dict(denoise_seconds=stages['DenoisingStage']/1000,
                                            decode_seconds=stages['DecodingStage']/1000,
                                            denoise_step_sum_seconds=sum(step['duration_ms'] for step in perf['denoise_steps_ms'])/1000))
    summaries = []
    for (quality,prompt), samples in groups.items():
        summaries.append(dict(quality=quality,prompt=prompt,n=len(samples),
                              mean_denoise_seconds=statistics.mean(row['denoise_seconds'] for row in samples),
                              median_denoise_seconds=statistics.median(row['denoise_seconds'] for row in samples),
                              mean_decode_seconds=statistics.mean(row['decode_seconds'] for row in samples),
                              raw=samples))
    rows.append(dict(configuration=arm['configuration'],source_sha=arm['source_sha'],diagnostic_stages=summaries,
                     diagnostic_full_process_peak_gpu0_memory_mib=max(memory_mib),
                     memory_scope='NVML samples across loading, warmup, all quality requests and profile; not an unprofiled serving-only estimate.',
                     gpu_samples_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                     launch_to_ready_seconds=arm['launch_to_ready_seconds'],
                     warmup_denoising_wall_seconds=arm['warmup_denoising_wall_seconds'],
                     actual_warmup_captures=arm['actual_warmup_captures'],serving_requests=arm['serving_requests'],
                     serving_capture_calls=arm['serving_capture_calls'],serving_replay_calls=arm['serving_replay_calls']))
result = dict(passed=True,source_sha=head,final_primary_admitted=True,primary_performance_result=False,
              raw_output_count=sum(len(row['outputs']) for row in data['raw']['rows']),rows=rows,
              all_same_quality_main_outputs_and_all_frames_exact=True,actual_fused_LN_path_verified=True,
              actual_main_PR_representative_frames_reviewed=True,
              initial_CPU_import_failure_retained='evidence/block-runs/sana-video-ln-final-tests1/cpu_compile.log',
              evidence_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limitations='Diagnostic runs use stage synchronization and an observer; their times are not primary serving results. Existing BCG compatibility only, no new BCG benefit claim. Exact/high use uncaptured quality-specific eager paths. Representative video frames were reviewed, not full motion. Chinese crossOmni PSNR failure retained. Other shapes/batches/concurrency/2GPU unrun.')
out = evidence/'preflight/sana-video-ln-final-compatibility1-admission.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='rows'},indent=2))
