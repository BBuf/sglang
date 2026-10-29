"""Export reproducible final-head evidence outside the production code branch."""
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
destination = root/'artifacts/sana-video-depthwise-glu'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert read(evidence/'preflight/sana-video-depthwise-final-primary-cleanup1-admission.json')['passed']
assert read(evidence/'preflight/sana-video-depthwise-final-primary1-raw-audit.json')['passed']
assert read(evidence/'preflight/sana-video-depthwise-final-compatibility1-admission.json')['passed']
assert read(evidence/'preflight/sana-video-depthwise-final-summary1.json')['source_sha']=='f248594caca5c533c94612b6848ff3512ed98e08'
assert read(evidence/'preflight/sana-video-depthwise-final-profile1.json')['passed']
assert read(evidence/'preflight/sana-video-omni-stages1-admission.json')['passed']
for name in ['sana-video-native1', 'sana-video-depthwise-qualification1', 'sana-video-depthwise-final-primary1', 'sana-video-omni-stages1']:
    receipt = read(evidence/'checkpoints'/f'{name}-local-verification.json')
    assert receipt.get('all_member_hashes_verified') or receipt.get('all_member_hashes_equal')
media = evidence/'media/sana-video-depthwise-final-f248594c'
assert read(media/'visual-review.json')['passed']
worst = evidence/'media/sana-video-native-cross-worst1'
worst_review = read(worst/'visual-review.json')
assert worst_review['passed'] and not worst_review['quality_passed']
assert sha(root/worst_review['panel']) == worst_review['panel_sha256']
destination.mkdir(exist_ok=False)
mapping, omitted = {}, []


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def export(source):
    relative = str(source.relative_to(root))
    digest = sha(source)
    target = destination/relative
    if source.suffix in ['.mp4', '.png']:
        target = destination/'media/by-sha256'/(digest+source.suffix)
        if not target.exists():
            copy(source, target)
        mapping[relative] = dict(public_path=str(target.relative_to(destination)), sha256=digest)
    elif source.suffix in ['.pt', '.bundle']:
        omitted.append(dict(path=relative, bytes=source.stat().st_size, sha256=digest,
                            reason='Binary tensor snapshots or Git transfer bundles remain in verified local archives; numerical audits and reproduction code are public.'))
    elif source.suffix == '.log' and source.stat().st_size > 256*1024:
        target = target.with_name(target.name+'.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(source.read_bytes(), compresslevel=6, mtime=0))
        mapping[relative] = dict(public_path=str(target.relative_to(destination)), sha256=digest,
                                 encoding='gzip; losslessly decompress to recover original bytes')
    else:
        assert source.stat().st_size < 100*1024**2, source
        copy(source, target)


for area in ['qualification', 'preflight', 'diagnostics', 'profiles', 'block-runs', 'benchmarks',
             'history', 'plans', 'runner-logs', 'downloads', 'harness-versions', 'reproduction']:
    for item in (evidence/area).glob('*sana-video*'):
        if 'sana-video-ln' in item.name or 'sana-video-eager-ln' in item.name:
            continue
        for source in ([item] if item.is_file() else sorted(item.rglob('*'))):
            if source.is_file() and '__pycache__' not in source.parts:
                export(source)
for source in media.iterdir():
    if source.is_file():
        copy(source, destination/'media'/source.name)
for folder in [evidence/'media/sana-video-native-cross-frames', worst]:
    for source in folder.iterdir():
        if source.is_file():
            copy(source, destination/source.relative_to(root))
for number in [34928, 35961, 38529]:
    source = evidence/'history'/f'sana-pr-{number}-full.json'
    copy(source, destination/source.relative_to(root))
video_rows = []
for row in read(media/'manifest.json')['rows']:
    for video in row['inputs']:
        source = root/video['video']
        assert sha(source) == video['sha256']
        target = destination/'media/videos'/f'{row["prompt"]}-{video["arm"].lower()}.mp4'
        copy(source, target)
        video_rows.append(dict(prompt=row['prompt'], arm=video['arm'], source_sha=video['source_sha'],
                               original_path=video['video'], public_path=str(target.relative_to(destination)),
                               sha256=video['sha256']))
for source in (root/'scripts').glob('*sana_video*'):
    if source.suffix in ['.py', '.sh']:
        copy(source, destination/'reproduce'/source.name)
for name in ['diffusion_skill_env.py', 'preflight_ming_environment.py',
             'check_sana_video_depthwise_cpu_compile1.py', 'download_pinned_modelscope_ranges.py']:
    copy(root/'scripts'/name, destination/'reproduce'/name)
for source in (root/'inputs').glob('sana-video*'):
    copy(source, destination/'inputs'/source.name)
for source in (evidence/'checkpoints').glob('sana-video*.json'):
    if 'sana-video-ln' in source.name or 'sana-video-eager-ln' in source.name:
        continue
    copy(source, destination/'archive-manifests'/source.name)
copy(evidence/'history/cleanup-style-notes.md', destination/'evidence/history/cleanup-style-notes.md')
readme_source = evidence/'sana-video-depthwise-evidence-readme1.md'
readme = readme_source.read_text()
prepared = 'This file is prepared before final qualification. Publication is gated on the final-head primary/cleanup admission, compatibility admission, actual-frame review and verified archives. It does not certify pending runs.'
assert prepared in readme
readme = readme.replace(prepared, 'Final-head primary/cleanup, compatibility, actual-frame review and archive verification gates passed before this export. The evidence branch is separate from production code; it contains no model weights. Upstream review/CI and visible PR-page media are separate submission gates.')
readme = readme.replace('After publication, read', 'Read')
(destination/'README.md').write_text(readme)
for source in (root/'experiments/sana-video-depthwise-glu').rglob('*'):
    if source.is_file() and '__pycache__' not in source.parts:
        export(source)
copy(evidence/'sana-video-depthwise-final-performance-table1.md', destination/'performance-table.md')
copy(root/'matrix-draft.json', destination/'frozen-matrix-with-execution.json')
registrations = [json.loads(line) for line in (root/'shared-files.jsonl').read_text().splitlines() if line.strip()]
registrations = [row for row in registrations if row.get('head') == 'f248594caca5c533c94612b6848ff3512ed98e08']
assert len(registrations) == 5 and len({row['path'] for row in registrations}) == 5
(destination/'shared-file-registration.json').write_text(json.dumps(registrations, indent=2)+'\n')
copy(evidence/'preflight/hardware.txt', destination/'hardware.txt')
for arm, filename in [('sgl', 'sglang-pip-freeze-final.txt'), ('omni', 'omni-pip-freeze-final.txt')]:
    source = evidence/'preflight/container-logs'/filename
    # Retain editable-source and local-wheel entries as well as ordinary pins.
    # These exact records were checked before publication; filtering alone loses provenance.
    copy(source, destination/f'{arm}-pip-freeze.txt')
    pins = [line for line in source.read_text().splitlines()
            if re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', line)]
    (destination/f'{arm}-version-pins.txt').write_text('\n'.join(pins)+'\n')
for name, value in [('original-path-map.json', mapping), ('omitted-binary-files.json', omitted),
                    ('actual-main-pr-videos.json', video_rows)]:
    (destination/name).write_text(json.dumps(value, indent=2)+'\n')
manifest = {str(path.relative_to(destination)):dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in destination.rglob('*') if path.is_file()}
assert all(row['bytes'] < 100*1024**2 for row in manifest.values())
(destination/'evidence-files-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(files=len(manifest), bytes=sum(row['bytes'] for row in manifest.values()),
                      mapped_paths=len(mapping), omitted_binaries=len(omitted), main_PR_videos=len(video_rows))))
