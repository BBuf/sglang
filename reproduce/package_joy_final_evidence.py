"""Package audited Joy evidence on its independent public branch, without pushing."""
import hashlib
import json
import re
import shutil
from pathlib import Path

from analyze_joy_blocks import read

root = Path(__file__).resolve().parents[1]
evidence = root / 'evidence'
destination = root / 'artifacts/joy-image-gelu'

# Refuse partial publication: every required final audit must already exist.
for shape in ('square', 'wide'):
    primary = read(evidence / 'benchmarks' / f'joy-gelu-final-{shape}-paired1-analysis.json')
    assert primary['source_and_ownership_audit_pass'] and len(primary['blocks']) == 6
    assert all(x['all_pairs_faster'] for x in primary['paired'].values())
    assert read(evidence / 'profiles' / f'joy-gelu-final-{shape}-profile-audit.json')['final_head_and_raw_trace_audit_pass']
    assert read(evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-audit.json')['all_raw_audits_pass']
    assert read(evidence / 'media' / f'joy-gelu-final-4c135ea-{shape}' / 'render-review.json')['all_three_figures_inspected']


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def response_metrics(source, target):
    raw = source.read_bytes()
    data = json.loads(raw)
    data.pop('data', None)
    data['_public_export'] = dict(original_sha256=hashlib.sha256(raw).hexdigest(),
                                 omitted='Repeated base64 image data; original response and PNG retained in task archives.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + '\n')


def qualification(name, target):
    source = evidence / 'qualification' / name
    for filename in ('contract.json', 'results.json'):
        copy(source / filename, target / filename)
    for row in read(source / 'results.json'):
        request = Path(row['request']).name
        copy(source / request / 'request.json', target / request / 'request.json')
        response_metrics(source / request / 'response.json', target / request / 'response-metrics.json')
    gate = evidence / 'qualification' / (name + '-quality.json')
    if gate.exists():
        copy(gate, target / 'quality.json')


def block(name, config, area):
    source = evidence / 'benchmarks' / name
    target = destination / area / 'raw' / name
    for filename in ('metadata.json', 'summary.json', 'results.json', 'source-status.txt',
                     'quality.json', 'runtime-precision.jsonl', 'native-log-correlation.json', 'source.patch'):
        path = source / filename
        if path.exists():
            copy(path, target / filename)
    for row in read(source / 'results.json'):
        index = f'{row["index"]:02d}'
        for filename in ('request.json', 'gpu-samples.json', 'perf.json'):
            path = source / index / filename
            if path.exists():
                copy(path, target / index / filename)
        response_metrics(source / index / 'response.json', target / index / 'response-metrics.json')
    for path in (evidence / 'block-runs' / name).iterdir():
        if path.is_file() and path.suffix in ('.json', '.sh', '.log', '.patch'):
            copy(path, target / 'server' / path.name)
    arm = 'omni' if config.startswith('omni') else 'sgl'
    qualification(f'{arm}-joy-{name}', target / 'qualification')


plans = ['joy-gelu-v1-aa1', 'joy-gelu-final-wide-aa1', 'joy-gelu-final-cleanup1',
         'joy-gelu-final-square-paired1', 'joy-gelu-final-wide-paired1']
for name in plans:
    for suffix in ('plan', 'analysis'):
        copy(evidence / 'benchmarks' / f'{name}-{suffix}.json', destination / 'results' / f'{name}-{suffix}.json')
    plan = read(evidence / 'benchmarks' / f'{name}-plan.json')
    for index, config in enumerate(plan['order']):
        block(f'{name}-{index:02d}-{config}', config, 'results')
for shape in ('square', 'wide'):
    for source, target in [
        (evidence / 'benchmarks' / f'joy-gelu-final-{shape}-paired1-summary.json', destination / 'results'),
        (evidence / 'qualification' / f'joy-gelu-final-{shape}-paired1-quality-summary.json', destination / 'results'),
        (evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-plan.json', destination / 'diagnostics'),
        (evidence / 'benchmarks' / f'joy-gelu-final-{shape}-omni-denoise1-audit.json', destination / 'diagnostics'),
        (evidence / 'profiles' / f'joy-gelu-final-{shape}-profile-audit.json', destination / 'diagnostics')]:
        copy(source, target / source.name)
    source = evidence / 'profiles' / f'joy-gelu-final-{shape}-matched-profile-analysis.json'
    copy(source, destination / 'diagnostics' / source.name)
    block(f'joy-gelu-final-{shape}-omni-denoise1-00-omni_compile_profile', 'omni_compile_profile', 'diagnostics')
    for arm in ('main', 'pr'):
        name = f'joy-gelu-final-{shape}-profile1-{arm}'
        for path in (evidence / 'profiles' / name).glob('*.trace.json.gz'):
            copy(path, destination / 'diagnostics/profiles' / name / path.name)
        for path in (evidence / 'block-runs' / name).iterdir():
            if path.is_file() and path.suffix in ('.json', '.sh', '.log', '.patch'):
                copy(path, destination / 'diagnostics/profiles' / name / 'server' / path.name)
        qualification(f'sgl-{name}-high', destination / 'diagnostics/profiles' / name / 'qualification')
    media = evidence / 'media' / f'joy-gelu-final-4c135ea-{shape}'
    media_target = destination if shape == 'square' else destination / 'wide'
    for path in media.iterdir():
        if path.suffix in ('.json', '.png'):
            copy(path, media_target / path.name)
for path in (evidence / 'qualification').glob('joy-gelu-v4-*.json'):
    copy(path, destination / 'quality' / path.name)
for path in (evidence / 'qualification').glob('*joy-gelu-v4-*-quality.json'):
    copy(path, destination / 'quality' / path.name)
copy(evidence / 'benchmarks/joy-gelu-final-cleanup-retest-summary.json', destination / 'results/cleanup-retest-summary.json')
copy(evidence / 'benchmarks/joy-final-native-diagnostic-admission-amendment.json', destination / 'diagnostics/native-diagnostic-admission-amendment.json')
for path in (evidence / 'runner-logs').glob('joy-gelu-final-*'):
    if path.suffix in ('.json', '.log'):
        copy(path, destination / 'validation/runner-logs' / path.name)
for path in (evidence / 'runner-logs').glob('joy-gelu-v4-gpu-test*'):
    copy(path, destination / 'validation/runner-logs' / path.name)
for path in (root / 'inputs').glob('joy-workloads*.json'):
    copy(path, destination / 'inputs' / path.name)
for name in ('reference-teapot.png', 'reference-teapot-1792x512.png', 'reference-teapot-512x1792.png'):
    copy(root / 'inputs' / name, destination / 'inputs' / name)
for path in (root / 'scripts').glob('*joy*'):
    if path.suffix in ('.py', '.sh'):
        copy(path, destination / 'reproduce' / path.name)
copy(root / 'scripts/compare_moss_qualification.py', destination / 'reproduce/compare_moss_qualification.py')
for path in (evidence / 'downloads').glob('joy-*.json'):
    copy(path, destination / 'downloads' / path.name)
for path in (evidence / 'harness-versions').glob('joy-final-*.json'):
    copy(path, destination / 'validation/harness-versions' / path.name)
for path in (evidence / 'preflight').glob('joy-*.json'):
    copy(path, destination / 'validation' / path.name)
for pattern in ('joy-gelu*v4*.log', 'joy-gelu*docs*.log'):
    for path in (evidence / 'preflight').glob(pattern):
        copy(path, destination / 'validation' / path.name)
for arm, name in [('sgl', 'sglang-pip-freeze-final.txt'), ('omni', 'omni-pip-freeze-final.txt')]:
    packages = [line for line in (evidence / 'preflight/container-logs' / name).read_text().splitlines()
                if re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', line)]
    (destination / 'validation' / f'{arm}-version-pins.txt').write_text('\n'.join(packages) + '\n')
for name in ('joy-gelu-cleanup-review.md', 'joy-gelu-v4-final-diff-review.json'):
    copy(evidence / 'history' / name, destination / 'validation' / name)
manifest = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in destination.rglob('*') if path.is_file() and '.git' not in path.parts
            and path.name != 'evidence-files-sha256.json'}
(destination / 'evidence-files-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Staged {len(manifest)} files. Review and update README before committing or pushing.')
