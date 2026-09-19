"""Remove only the validated campaign-owned LongCat Turbo checkpoint."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess

root = Path('/campaign')
art = root/'artifacts/longcat-image-edit-turbo'
for name in ('longcat-combined-abba', 'longcat-combined-abba-r2',
             'longcat-quality-matrix', 'longcat-gelu-cat-tests',
             'longcat-gelu-cat-model-tests', 'longcat-combined-config-tests'):
    assert (root/f'{name}.exit').read_text().strip() == '0', name
evidence = json.loads((art/'final-evidence.json').read_text())
assert evidence['qualified']
assert len(evidence['quality_matrix']) == 8
assert evidence['graph_replay']['graph_launches'] == {'cudaGraphLaunch': 248}
for row in evidence['output_comparison']:
    if row['quality'] == 'lossless' and row['valid_performance_row']:
        assert row['pixel_exact'], row
repo = root/'candidate-longcat-gelu-cat'
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == '10410a20b6d2716c1489a3132dc79c775dc297bb'
subprocess.run(['git', '-C', str(repo), 'diff', '--quiet'], check=True)
subprocess.run(['git', '-C', str(repo), 'diff', '--cached', '--quiet'], check=True)
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
record = bench._cleanup_model_cache(root/'model-caches', root/'model-caches/longcat-image-edit-turbo-cycle',
    root/'artifacts/cache-cleanup.jsonl', 'longcat-image-edit-turbo', 'cycle',
    'Repeated native saved-request E2E, actual graph replay, full profiles, exact lossless images, quality applicability and kernel validation completed')
assert record['after'] == dict(file_count=0, weight_file_count=0, total_bytes=0), record
assert not (root/'model-caches/longcat-image-edit-turbo-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
