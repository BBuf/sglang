"""Export admitted isolated Joy repair evidence to a separate unpublished directory."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
destination = root/'artifacts/joy-input-lifetime-repair1-03368cf1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = read(evidence/'preflight/joy-input-lifetime-repair1-summary1.json')
assert summary['numerical_admission_passed'] and summary['source_sha'] == '03368cf1c325e526184b8fb223443e1ae931b13c'
media = evidence/'media/joy-input-lifetime-repair1-03368cf1'
manifest = read(media/'manifest.json')
review = read(media/'visual-review.json')
assert review['passed'] and review['all_12_panels_inspected']
assert review['manifest_sha256'] == digest(media/'manifest.json')
assert len(manifest['rows']) == 12
assert manifest['source_sha'] == summary['source_sha']
assert manifest['official_PR_head'] == 'b08c24e9630c63790c9f86b48d6c7fc10a0c4301'
assert manifest['repair_is_official_PR_head'] is False
assert summary['official_PR_current_head_qualified'] is False and summary['new_optimization_slot_assigned'] is False
failed_official = read(evidence/'preflight/joy-current-head-square-primary1-admission.json')
assert failed_official['primary_passed'] and not failed_official['cleanup_passed'] and not failed_official['passed']
plans = [read(evidence/'plans'/f'joy-input-lifetime-repair1-{shape}-{phase}1.json')
         for shape in ('square', 'wide') for phase in ('aa', 'primary')]
plans.append(read(evidence/'plans/joy-input-lifetime-repair1-diagnostics1.json'))
for plan in plans:
    assert read(evidence/'preflight'/(plan['id']+'-admission.json'))['passed']
    archive = read(evidence/'checkpoints'/(plan['id']+'-local-verification.json'))
    assert archive['all_member_hashes_verified'] and archive['all_members_extracted']
destination.mkdir(exist_ok=False)
mapping, omitted = {}, []


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def export(path):
    name = str(path.relative_to(root))
    target = destination/name
    sha = digest(path)
    if path.suffix == '.png':
        target = destination/'media/by-sha256'/(sha+'.png')
        if not target.exists():
            copy(path, target)
        mapping[name] = dict(public_path=str(target.relative_to(destination)), sha256=sha)
    elif path.name == 'response.json':
        value = read(path)
        if isinstance(value, dict) and 'data' in value:
            value.pop('data')
            value['_public_export'] = dict(original_sha256=sha,
                omitted='Repeated base64 image payload; actual output PNG is exported by content hash and exact response remains in verified archives.')
            target = target.with_name('response-metrics.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, indent=2)+'\n')
            omitted.append(dict(path=name, sha256=sha, bytes=path.stat().st_size,
                                public_metrics=str(target.relative_to(destination)), reason=value['_public_export']['omitted']))
        else:
            copy(path, target)
    elif path.suffix in ('.bundle', '.pt'):
        omitted.append(dict(path=name, sha256=sha, bytes=path.stat().st_size,
                            reason='Transfer bundle or tensor snapshot retained in local verified archives; source commits and audits are public.'))
    elif path.suffix == '.log' and path.stat().st_size > 256*1024:
        target = target.with_name(target.name+'.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(path.read_bytes(), compresslevel=6, mtime=0))
        mapping[name] = dict(public_path=str(target.relative_to(destination)), sha256=sha, encoding='gzip')
    else:
        assert path.stat().st_size < 100*1024**2, path
        copy(path, target)


selected = {
    evidence/'preflight/joy-current-head-square-primary1-admission.json',
    evidence/'preflight/joy-current-head-square-primary1-failure-review1.json',
    evidence/'preflight/joy-current-head-lifetime-probe1-audit.json',
    evidence/'preflight/joy-current-head-tests1-admission.json',
    evidence/'preflight/joy-current-head-failure-publication1.json',
}
for area in ('preflight', 'plans', 'block-runs', 'benchmarks', 'qualification', 'profiles', 'diagnostics',
             'reproduction', 'runner-logs', 'harness-versions', 'checkpoints'):
    for item in (evidence/area).glob('*joy-input-lifetime-repair1*'):
        selected.update([item] if item.is_file() else (p for p in item.rglob('*') if p.is_file()))
references = set()
for plan in plans:
    for name in plan['sha256']:
        relative = Path(name)
        if relative.parts[0] == 'outputs':
            selected.add(evidence/relative.relative_to('outputs'))
        else:
            selected.add(root/relative)
    for names in plan.get('quality_references', {}).values():
        references.update(names)
    for shape in plan.get('references', {}).values():
        for source in shape.values():
            references.update(source.values())
for name in references:
    folder = evidence/'qualification'/name
    assert folder.is_dir()
    selected.update(p for p in folder.rglob('*') if p.is_file())
for path in (root/'scripts').glob('*joy_input_lifetime_repair*.py'):
    selected.add(path)
for name in ('download_pinned_modelscope_ranges.py', 'diffusion_skill_env.py', 'preflight_diffusion_current_head1.py'):
    selected.add(root/'scripts'/name)
for path in (evidence/'downloads').glob('joy-*.json'):
    selected.add(path)
for path in (root/'inputs').glob('joy-workloads*.json'):
    selected.add(path)
for path in (root/'inputs').glob('reference-teapot*.png'):
    selected.add(path)
for path in sorted(selected):
    if '__pycache__' in path.parts:
        continue
    if path.name.endswith('.tar.gz'):
        # Archive bytes are kept locally; each public manifest preserves all member hashes.
        omitted.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                            sha256=digest(path),
                            reason='Verified complete archive retained locally; member manifest and exported evidence are public.'))
    else:
        export(path)
for path in media.iterdir():
    if path.is_file():
        copy(path, destination/'media'/path.name)
copy(evidence/'joy-input-lifetime-repair1-performance-table1.md', destination/'performance-table.md')
copy(root/'matrix-draft.json', destination/'frozen-matrix-with-execution.json')
for name in ('hardware.txt', 'container-logs/sglang-pip-freeze-final.txt', 'container-logs/omni-pip-freeze-final.txt'):
    copy(evidence/'preflight'/name, destination/'environment'/Path(name).name)
readme = '''# Joy input-lifetime repair evidence

This package validates the isolated repair03368cf1c325e526184b8fb223443e1ae931b13c, based on merged PR47022 headb08c24e9630c63790c9f86b48d6c7fc10a0c4301. It does not validate the official b08c head. The frozen official-head square cleanup test failed; its full public explanation and actual images remain at https://github.com/sgl-project/sglang/pull/47022 and its separate evidence commitf5743fe06a9585b6e0e0e3f90d6eb6ab6df381eb. The retained failure admission is included here. No qualification count or new independent optimization slot is assigned by this evidence export.

The repair releases the temporary normalized input before the output projection allocates. Actual red/green SM120 regression logs demonstrate the lifetime bug before and after the minimal fix; the original numerical, quality restoration, fallback and dynamic-LoRA checks remain. A separate archived lifetime probe identifies one retained BF16 input of56/64MiB at7168/8192tokens. That probe does not attribute every full-model NVML or timing difference to this one allocation.

[Measurements](performance-table.md) use frozen mainaf21e2b989e482cfdce39689126562f2209d53ac, originally tested optimization4c135ea92d6b4cbb0ead69bb39976f7dbe995455, isolated repair03368cf1c325e526184b8fb223443e1ae931b13c and native Omni817d4c17feb702289a2d2b2cbacfd594c262b7a7. Omni retains regional compilation and CUDNN_ATTN. The table's candidate_current arm means the isolated repair, never official headb08c. Fresh SYNC0 primary requests are separated from synchronized denoise/profile diagnostics; historical SYNC1 timings and the prior failed official-head experiment are not pooled. All four AA/primary archives and the diagnostic archive were independently verified. Every original sample remains; gates were not relaxed.

The recorded remote root is `/data/xyzhang/diffusion-rtx6000-20261026`, mounted at `/task` in the owned containers. Reproduce in a fresh directory with the exact source commits and scripts at the plan-specified paths. Verify the original checkpoint revision4b41fb25d961f37668750178ccbb380da326201c against its36-file manifest. Preserve native inputs, precision, backends, compilation flags, GPU0 ownership, warmups and process pairs. Launchers reject stale source and occupied GPUs. Actual regression and AA admission precede primary measurement; diagnostics require both primary admissions. Never overwrite runs or bypass failed gates.

[Media provenance](media/manifest.json) identifies the actual main/repair requests, and [visual review](media/visual-review.json) records inspection. Twelve panels show three high prompts and one returned-lossless example per square/wide/tall shape. All source PNGs are deduplicated by hash; `original-path-map.json` retains their original paths. Large logs are losslessly gzipped. Repeated response base64 is omitted only in public response-metrics files, with original hashes in `omitted-content.json`; source PNGs remain public and full responses remain in verified local archives. Pinned historical native-reference directories reproduce quality checks; they are not fresh timing data.

Weights, Git transfer bundles and tensor snapshots are excluded. `evidence-files-sha256.json` pins the public export. Environment records contain exact editable-source/local-wheel provenance; they do not establish a portable container. Tall is a quality/shape check only. There is no new BCG support/net-win, compiled-GPU performance or full-model LoRA claim. Publication and official-head qualification remain separate decisions.

'''
(destination/'README.md').write_text(readme)
for name, value in [('original-path-map.json', mapping), ('omitted-content.json', omitted)]:
    (destination/name).write_text(json.dumps(value, indent=2)+'\n')
files = {str(p.relative_to(destination)):dict(bytes=p.stat().st_size, sha256=digest(p))
         for p in destination.rglob('*') if p.is_file()}
assert all(x['bytes'] < 100*1024**2 for x in files.values())
(destination/'evidence-files-sha256.json').write_text(json.dumps(files, indent=2)+'\n')
print(json.dumps(dict(files=len(files), bytes=sum(x['bytes'] for x in files.values()),
                      references=len(references), panels=12, publication_pending=True)))
