"""Stage complete audited evidence outside the code PR; never publish automatically."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
DEST=ROOT/'artifacts/qwen21-bf16-gate-up'
def read(path):return json.loads(path.read_text())

assert read(E/'preflight/qwen21-gate-up-v4-cleanup-admission.json')['status']=='accepted_with_explicit_fresh_process_screen_failures'
compatibility=read(E/'qualification/qwen21-gate-up-v4-compatibility1-raw-audit.json')
assert compatibility['status']=='pass' and compatibility['no_serving_capture']
assert compatibility['all_RGBA_byte_exact_corresponding_main'] and compatibility['quality_transitions_pass']
assert read(E/'profiles/qwen21-gate-up-v4-final-1024-representative-profile-audit.json')['all_dispatch_pass']
for size in [512,1024]:
    run=f'qwen21-gate-up-v4-final-{size}-compare1'
    primary=read(E/'preflight'/f'{run}-audit.json')
    admission=read(E/'preflight'/f'{run}-admission.json')
    assert primary['status']=='raw_contract_pass' and primary['source_and_gpu_ownership_pass']
    assert admission['all_cases_accepted']
    for case in primary['checks'].values():
        assert all(x>1 for comparison in case['comparisons'].values() for x in comparison['paired_ratios'])
        if size==512:
            assert case['comparisons']['main_over_final']['latency_reduction_percent']>=1.5
    assert read(E/'qualification'/f'{run}-raw-audit.json')['source_and_native_backend_pass']
    assert read(E/'preflight'/f'qwen21-gate-up-v4-final-{size}-diagnostics1-audit.json')['all_raw_audits_pass']
    assert read(E/'media'/f'qwen21-gate-up-final-{size}'/'visual-review.json')['both_panels_inspected']

DEST.mkdir(parents=True,exist_ok=True)
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)

def export_response(source,target):
    data=read(source)
    data.pop('data',None)
    data['_public_export']=dict(original_response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),omitted='Duplicate base64 PNG data; original responses and every PNG remain in task archives. Output hashes are in results.json and raw audit manifests.')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,indent=2)+'\n')

def export_folder(source,target,include_png=False):
    for path in source.rglob('*'):
        if not path.is_file():continue
        dest=target/path.relative_to(source)
        if path.name=='response.json':export_response(path,dest.with_name('response-metrics.json'))
        elif path.suffix=='.png':
            if include_png:copy(path,dest)
        elif path.suffix in ['.json','.jsonl','.txt','.sh','.log','.patch','.gz']:
            copy(path,dest)

# Includes original failed/inconclusive cleanup screens, AA and configuration screens.
runs=['qwen21-gate-up-main-high-aa1','qwen21-gate-up-v1-abba1','qwen21-main-high-512-aa1',
      'qwen21-gate-up-v3-512-abba1','qwen21-gate-up-v4-512-cleanup1','qwen21-gate-up-v4-512-cleanup2',
      'qwen21-cleanup-within-512-1','qwen21-cleanup-within-1024-1','qwen21-omni-valid-variant-screen1']
runs += [f'qwen21-gate-up-v4-final-{size}-{kind}' for size in [512,1024] for kind in ['compare1','diagnostics1']]
for run in runs:
    plan=read(E/'plans'/f'{run}.json')
    copy(E/'plans'/f'{run}.json',DEST/'plans'/f'{run}.json')
    for index,config in enumerate(plan['order']):
        name=f'{run}-{index:02d}-{config}'
        export_folder(E/'block-runs'/name,DEST/'raw'/name/'server')
        spec=plan['configurations'][config]
        for task in spec.get('tasks',plan['tasks']):
            export_folder(E/'benchmarks'/f'{name}-{task}',DEST/'raw'/name/task/'benchmark')
            qual=f'{spec["arm"]}-qwen21-base-{task}-{name}-{plan["quality"]}'
            export_folder(E/'qualification'/qual,DEST/'raw'/name/task/'qualification')
            for suffix in ['-quality.json','-sgl-exact-quality.json']:
                path=E/'qualification'/(qual+suffix)
                if path.exists():copy(path,DEST/'raw'/name/task/path.name)

for area in ['preflight','qualification','history','runner-logs','plans']:
    for path in (E/area).glob('qwen21*'):
        if path.is_file() and path.suffix in ['.json','.md','.log','.patch','.py']:
            copy(path,DEST/'audits'/area/path.name)
for name in ['qwen21-gate-up-v4-compatibility1','qwen21-gate-up-v4-final-1024-diagnostics1']:
    for folder in (E/'profiles').glob(name+'*'):
        if folder.is_dir():export_folder(folder,DEST/'profiles'/folder.name)
for path in (E/'profiles').glob('qwen21*.json'):copy(path,DEST/'audits/profiles'/path.name)
for folder in (E/'diagnostics').glob('qwen21*'):
    if folder.is_dir():export_folder(folder,DEST/'diagnostics'/folder.name)
for folder in (E/'block-runs').glob('qwen21-gate-up-v4-compatibility1-*'):
    export_folder(folder,DEST/'compatibility'/folder.name)
for folder in (E/'qualification').glob('sgl-qwen21-base-*-qwen21-gate-up-v4-compatibility1-*'):
    if folder.is_dir():export_folder(folder,DEST/'compatibility/qualification'/folder.name)
for size in [512,1024]:
    source=E/'media'/f'qwen21-gate-up-final-{size}'
    for path in source.iterdir():
        if path.is_file():copy(path,DEST/'media'/str(size)/path.name)
    # Keep all unique main/final source images, not only the rendered panels.
    for task in ['t2i','edit']:
        for index,config in enumerate(['main','final']):
            name=f'qwen21-gate-up-v4-final-{size}-compare1-{index:02d}-{config}'
            source=E/'qualification'/f'sgl-qwen21-base-{task}-{name}-high'
            for row in read(source/'results.json')[:3]:
                path=source/Path(row['request']).name/'output.png'
                copy(path,DEST/'media/source-images'/str(size)/task/config/(path.parent.name+'.png'))
for path in (ROOT/'scripts').glob('*qwen21*'):
    if path.suffix in ['.py','.sh']:copy(path,DEST/'reproduce'/path.name)
for name in ['analyze_joy_gelu_profiles.py','analyze_joy_blocks.py']:
    copy(ROOT/'scripts'/name,DEST/'reproduce'/name)
for path in (E/'harness-versions').glob('qwen21*'):
    if path.is_file():copy(path,DEST/'audits/harness-versions'/path.name)
copy(ROOT/'matrix-draft.json',DEST/'frozen-matrix-with-execution.json')
for path in (ROOT/'inputs').glob('qwen21*'):copy(path,DEST/'inputs'/path.name)
copy(ROOT/'inputs/reference-teapot.png',DEST/'inputs/reference-teapot.png')
copy(E/'downloads/qwen21-modelscope-download-manifest.json',DEST/'inputs/checkpoint-manifest.json')
for arm,filename in [('sgl','sglang-pip-freeze-final.txt'),('omni','omni-pip-freeze-final.txt')]:
    pins=[line for line in (E/'preflight/container-logs'/filename).read_text().splitlines() if re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line)]
    (DEST/f'{arm}-version-pins.txt').write_text('\n'.join(pins)+'\n')
manifest={str(p.relative_to(DEST)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in DEST.rglob('*') if p.is_file() and '.git' not in p.parts and p.name!='evidence-files-sha256.json'}
assert all(item['bytes']<100*1024*1024 for item in manifest.values())
(DEST/'evidence-files-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Staged {len(manifest)} files. Review README, source hashes and media before committing or publishing.')
