"""Build a separate evidence repository; this script does not publish it."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'evidence/public/nomas-mean-37999'
assert not out.exists() or (sys.argv[1:] == ['--refresh'] and not (out/'.git').exists())
report = json.loads((root/'evidence/performance/nomas-mean-final12.json').read_text())
assert report['all80_outputs_exact']
assert all(x['gain_exceeds_1_5pct_and_noise'] for x in report['comparisons']['final'].values())
assert all(x['cleanup_change_within_noise'] for x in report['comparisons']['cleanup'].values())
out.mkdir(parents=True,exist_ok=True)
included = {}


def copy(path):
    path = Path(path)
    rel = path.relative_to(root)
    if str(rel) in included:
        return
    assert path.is_file() and not path.is_symlink(), path
    assert path.stat().st_size < 95*1024**2, path
    target = out/rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if rel == Path('manifest.json'):
        data = json.loads(path.read_text())
        for key in ('thread_id','host','ssh_user','ssh_port','hostname','reservation'):
            data.pop(key, None)
        data['public_copy_note'] = 'Login endpoint and chat identifiers omitted. Original retained privately.'
        data['original_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        target.write_text(json.dumps(data,indent=2)+'\n')
    else:
        shutil.copyfile(path,target)
    if target.suffix in ('.json','.jsonl','.log','.txt','.md','.py'):
        text = target.read_text(errors='replace')
        assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|https?://[^\s/:]+:[^\s/@]+@)',text), rel
    included[str(rel)] = dict(path=str(rel),bytes=target.stat().st_size,
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest())


for name in ['manifest.json','scope-frozen.json','scope-frozen.sha256','results-matrix.json',
             'evidence/history/cleanup-style.md','evidence/quality/nomas-final-media-review16.json']:
    copy(root/name)
for folder, pattern in [('inputs','nomas*.json'),('evidence/performance','nomas*.json'),
                        ('evidence/history','nomas*'),('evidence/preflight','nomas*')]:
    for path in (root/folder).glob(pattern):
        if path.is_file() and not any(s in path.name for s in ['cdn','transfer','staging','direct','launch','owned-environments']):
            copy(path)
for name in ['nomas-native-workloads01.json','nomas-weight-staging01.json']:
    copy(root/'inputs'/name)
for path in (root/'inputs/nomas-metadata01').rglob('*'):
    if path.is_file() and path.suffix in ('.json','.txt','.jinja'):
        copy(path)
# The original materialization script checks this prior task-owned idle receipt.
copy(root/'evidence/preflight/h3-image-row-overlap-pilot02-status.json')
scripts = ['build_nomas_video_payload.py','nomas_cpu_contract01.py','nomas_action_mask_cpu01.py',
           'materialize_nomas_common01.py','nomas_checkpoint_key_audit02.py','materialize_nomas_keys02.py',
           'audit_nomas_native01.py','audit_nomas_native02.py','audit_nomas_native09.py',
           'analyze_nomas_profile05.py','analyze_nomas_profile11.py','analyze_nomas_measurements12.py',
           'check_nomas_fallback15.py','prepare_nomas_mean_media11.py','run_nomas_noise10.py',
           'run_nomas_measurements12.py','run_nomas_measurements14.py','run_nomas_validation11.py',
           'run_nomas_scope15.py','continue_nomas_scope16.py','prepare_nomas_public16.py']
for name in scripts:
    copy(root/'scripts'/name)
for folder in ['nomas-native-visits','nomas-native-visits02','nomas-native-observer01',
               'nomas-native-observer15','nomas-quiet-visits','nomas-measure-visits12',
               'nomas-profile-visits','nomas-cleanup-visits11','nomas-cleanup-long-visits13',
               'nomas-cleanup-profile11','nomas-vae-mean-observer07','nomas-scope-visits15','nomas-scope-observer15']:
    for path in (root/'scripts'/folder).glob('*.py'):
        copy(path)
copy(root/'scripts/h3-native-visits/diffusion_skill_env.py')

quality_folders = ['nomas-49-native02','nomas-121-native09','nomas-49-main-cleanup11',
                   'nomas-121-main-cleanup11','nomas-49-high-main-cleanup11']
for name in quality_folders:
    for path in (root/'evidence/quality'/name).rglob('*'):
        if path.is_file() and path.suffix in ('.json','.png','.log','.py'):
            copy(path)
for name in ['nomas-49-main-cleanup11','nomas-121-main-cleanup11','nomas-49-high-main-cleanup11']:
    q = json.loads((root/'evidence/quality'/name/'summary.json').read_text())
    assert all(p['artifact_exact'] and p['decoded_rgb_exact'] for p in q['pairs'])
    for p in q['pairs']:
        for path in p['paths']:
            copy(root/path)

run_names = ['nomas-49-omni-native01','nomas-49-sgl-native01','nomas-49-sgl-native02',
             'nomas-121-omni-native09','nomas-121-sgl-native09','nomas-49-sgl-mean-quality11',
             'nomas-121-sgl-mean-quality13','nomas-49-high-main-scope15','nomas-49-high-PR-scope16',
             'nomas-49-sgl-profile05','nomas-49-sgl-mean-profile11']
run_names += list(report['visits'])
run_names += [p.name for p in (root/'runs').glob('*nomas*noise10*')]
for name in run_names:
    folder = root/'runs'/name
    assert folder.is_dir(), folder
    for path in folder.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.log','.txt'):
            copy(path)
for name in ['nomas-49-sgl-profile05','nomas-49-sgl-mean-profile11']:
    for path in (root/'runs'/name/'profiles').glob('*.trace.json.gz'):
        copy(path)
    copy(root/'evidence/profiles'/name/'raw-costs.json')
copy(root/'evidence/profiles/nomas-mean-main-cleanup11-digest.json')

rows=[]
for prompt,cell in report['comparisons']['final'].items():
    rows.append(f"| {prompt} | {cell['A']['p50_seconds']:.4f} / {cell['A']['p95_seconds']:.4f} | "
                f"{cell['B']['p50_seconds']:.4f} / {cell['B']['p95_seconds']:.4f} | "
                f"{cell['paired_geometric_speedup']:.6f}x | {cell['latency_reduction_pct']:.3f}% | "
                f"{cell['observed_descriptive_noise_bound_pct']:.3f}% |")
table='\n'.join(['| Prompt | Main E2E p50 / p95 s | PR E2E p50 / p95 s | Paired speedup | Reduction | Observed noise |',
                '|---|---:|---:|---:|---:|---:|',*rows])
cross=[]
for frames in (49,121):
    q=json.loads((root/f'evidence/quality/nomas-{frames}-{"native02" if frames==49 else "native09"}/summary.json').read_text())
    for cell in q['pairs']:
        if not cell['name'].startswith('cross-'):
            continue
        assert not cell['numeric_passed']
        cross.append(f"| {cell['name'][6:]} | {cell['minimum_ssim']:.6f} | {cell['minimum_psnr_db']:.3f} | fail |")
cross_table='\n'.join(['| Native Omni / SGL main cell | Minimum all-frame SSIM | Minimum PSNR dB | Frozen gate |','|---|---:|---:|---|',*cross])
startup={name:{k:v[k] for k in ['source_sha','launch_to_ready_seconds','warmup_saved_request_seconds','serial_requests_per_second','sampled_total_device_peak_MiB']} for name,v in report['visits'].items() if 'final12' in name}
(out/'startup-and-throughput.json').write_text(json.dumps(startup,indent=2)+'\n')
(out/'benchmark-table.md').write_text(table+'\n')
(out/'cross-framework-table.md').write_text(cross_table+'\n')
(out/'artifact-manifest.json').write_text(json.dumps(list(included.values()),indent=2)+'\n')
print(json.dumps(dict(files=len(included),bytes=sum(p['bytes'] for p in included.values()),path=str(out))))
