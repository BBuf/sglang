"""Require qualified native evidence, then remove only the owned FastH3 weights."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess

root = Path('/campaign')
art = root/'artifacts/fasth3'
for name in ('admit-fasth3-scoped', 'fasth3-profile-post', 'fasth3-baseline-triage',
             'fasth3-swiglu-tests', 'fasth3-swiglu-existing-tests', 'fasth3-swiglu-microbench',
             'fasth3-swiglu-abba', 'fasth3-swiglu-post'):
    assert (root/f'{name}.exit').read_text().strip() == '0', name
for repo, sha in [('baseline', '993d1fccbaafe3e79d91567d2fc1d665cc94fa50'),
                 ('candidate-fasth3-swiglu', '8ec2a429aeb1999303165925859b47a30a7aaf37')]:
    assert subprocess.check_output(['git', '-C', str(root/repo), 'rev-parse', 'HEAD'], text=True).strip() == sha
    subprocess.run(['git', '-C', str(root/repo), 'diff', '--quiet'], check=True)
    subprocess.run(['git', '-C', str(root/repo), 'diff', '--cached', '--quiet'], check=True)
summary = json.loads((art/'swiglu-r1-detailed-summary.json').read_text())
for metric in ('engine_e2e_s', 'client_e2e_s'):
    assert summary['means'][metric]['reduction_pct'] >= 1.5, summary['means']
comparison = json.loads((art/'swiglu-output-comparison/comparison.json').read_text())
assert comparison['all_exact'] and len(comparison['runs']) == 4
profile = {}
outputs = []
for arm, label in [('baseline', 'fp8-scoped-baseline-profile'), ('candidate', 'fp8-swiglu-profile')]:
    evidence = json.loads((art/label/'traces/denoise-step2-evidence.json').read_text())
    assert evidence['crossing_kernel_count'] == 0 and evidence['swiglu_calls'] == 50
    profile[arm] = evidence
    result = json.loads((art/label/'result.json').read_text())
    outputs.extend(result['output_sha256'])
outputs.extend(r['artifact_sha256'] for r in comparison['runs'])
assert len(set(outputs)) == 1, outputs
assert profile['candidate']['swiglu_kernel_count'] == 50
assert profile['baseline']['swiglu_kernel_count'] == 100
assert profile['candidate']['swiglu_gpu_ms'] < profile['baseline']['swiglu_gpu_ms']
micro = json.loads((art/'swiglu-committed-microbench.json').read_text())
assert all(r['exact'] and r['input_preserved'] for r in micro['rows'])
final = dict(qualified=True, submitted_pr=None,
             source_checkpoint='FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree@5ea076f35b84da4c3c82217112fa733d8eea2ae1',
             overlay='kevin-mi/FastH3-4step-Preview-overlay@f769cb8001dae335089de7b250364335bc7cb183',
             benchmark=summary, output_comparison=comparison, profile=profile, microbenchmark=micro,
             profiled_and_unprofiled_mp4_byte_identical=True,
             bcg='Native pipeline disabled BCG; actual probe invalid and excluded',
             tests='Four CUDA cases plus 11 existing FastH3/VSA cases passed on H200',
             note='Profiles are diagnostic. VDN screening numbers are not FastH3 E2E evidence.')
(art/'final-evidence.json').write_text(json.dumps(final, indent=2))
lock = (root/'gpu.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True)
selected = {line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
assert not any(line.split(',')[0].strip() in selected for line in processes.splitlines()), processes
helper = root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec = importlib.util.spec_from_file_location('campaign_bench', helper)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
record = bench._cleanup_model_cache(root/'model-caches', root/'model-caches/fasth3-cycle',
                                    root/'artifacts/cache-cleanup.jsonl', 'fasth3', 'cycle',
                                    'Qualified native FP8 eager SwiGLU optimization with ABBA, paired profiles, microbench and exact decoded video/audio evidence; publication follows')
assert record['after'] == dict(file_count=0, weight_file_count=0, total_bytes=0), record
assert not (root/'model-caches/fasth3-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record, indent=2))
print(json.dumps(dict(means=summary['means'], cleanup=record), indent=2), flush=True)
