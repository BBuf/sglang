"""Prepare evidence for the isolated repair without qualifying the merged PR head."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--artifact-commit', required=True)
args = parser.parse_args()
assert re.fullmatch('[0-9a-f]{40}', args.artifact_commit)
root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
artifacts = root/'artifacts/joy-input-lifetime-repair1-03368cf1'
media = evidence/'media/joy-input-lifetime-repair1-03368cf1'
head = '03368cf1c325e526184b8fb223443e1ae931b13c'
official = 'b08c24e9630c63790c9f86b48d6c7fc10a0c4301'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert subprocess.check_output(['git', '-C', str(artifacts), 'rev-parse', 'HEAD'], text=True).strip() == args.artifact_commit
assert not subprocess.check_output(['git', '-C', str(artifacts), 'status', '--porcelain'])
summary = read(evidence/'preflight/joy-input-lifetime-repair1-summary1.json')
diagnostics = read(evidence/'preflight/joy-input-lifetime-repair1-diagnostics1-admission.json')
failed = read(evidence/'preflight/joy-current-head-square-primary1-admission.json')
tests = read(evidence/'preflight/joy-input-lifetime-repair1-tests-admission.json')
manifest = read(media/'manifest.json')
review = read(media/'visual-review.json')
assert summary['source_sha'] == diagnostics['source_sha'] == tests['source_sha'] == head
assert summary['numerical_admission_passed'] and diagnostics['passed'] and tests['passed']
assert not summary['official_PR_current_head_qualified'] and not summary['new_optimization_slot_assigned']
assert not failed['passed'] and not failed['cleanup_passed']
assert manifest['source_sha'] == head and manifest['official_PR_head'] == official
assert not manifest['repair_is_official_PR_head'] and len(manifest['rows']) == 12
assert review['passed'] and review['all_12_panels_inspected']
assert review['manifest_sha256'] == digest(media/'manifest.json')
table_path = evidence/'joy-input-lifetime-repair1-performance-table1.md'
assert digest(table_path) == summary['table_sha256'] == digest(artifacts/'performance-table.md')
base = f'https://github.com/BBuf/sglang/blob/{args.artifact_commit}'
raw = f'https://raw.githubusercontent.com/BBuf/sglang/{args.artifact_commit}'


def link(label, path):
    assert (artifacts/path).is_file(), path
    return f'[{label}]({base}/{path})'


body = [
    '<!-- joy-isolated-input-lifetime-repair-20261030 -->',
    '## Isolated input-lifetime repair: validated separately from the merged head', '',
    f'This supplement tests isolated repair `{head}`, whose parent is merged PR head `{official}`. The merged head still has the failed square cleanup result reported above. These measurements do not change that head or turn its failure into a pass. No new independent optimization is counted.', '',
    'The repair restores assignment to `x` before the output projection, releasing the temporary normalized input before `fc_out` allocates. It preserves the cleaned-up projection name, eligibility checks, fused arithmetic and output wrapper. The existing regression test now checks that a temporary input has been released at entry to `fc_out`, while retaining numerical, restoration, compile-frontend, fallback and dynamic-LoRA checks.', '',
    'The actual SM120 regression failed on the parent with the input-retention assertion and passed on the repair, one test with zero skips in each run. A separate production-MLP probe measured one extra BF16 input on the parent: 56/64 MiB for 7168/8192 tokens, with identical outputs and kernel dispatch. This does not establish that this allocation caused the entire earlier 138 MiB NVML difference or all of the measured timing difference.', '',
    'Whole-model measurements use frozen SGLang main `af21e2b989e482cfdce39689126562f2209d53ac`, original tested optimization `4c135ea92d6b4cbb0ead69bb39976f7dbe995455`, the isolated repair, and native vLLM-Omni `817d4c17feb702289a2d2b2cbacfd594c262b7a7`. Same RTX PRO 6000 Blackwell Server GPU0; GPU1 idle. Checkpoint `4b41fb25d961f37668750178ccbb380da326201c`, BF16 DiT/encoder/VAE, seed 1101, 30 steps, CFG 4, batch/concurrency 1. Omni retains regional compilation and CUDNN_ATTN.', '',
    'Each shape has two fresh main A/A processes and the fixed 12-process order main–original–repair–Omni–original–repair–repair–original–Omni–repair–original–main. Each process runs four qualification requests, two excluded warmups and four measured requests. All primary SGLang timings use SYNC0 without profiler or observer. Historical timings and the failed official-head run are preserved separately. Cleanup gates remain mean regression no greater than fresh A/A noise and four-pair log-t95 upper regression below 0.1%; all main/repair and Omni/repair pairs must improve, and main E2E reduction must exceed 1.5% and A/A noise.', '',
    table_path.read_text().strip(), '',
    'The separate SYNC1 diagnostics run lossless → high → lossless for square, wide and tall shapes. All 76 outputs and all 36 main/repair quality comparisons are audited. Four separate profiles establish that the fused path actually runs. Tall is a quality/shape check only; this is not new BCG support, compiled-GPU performance, full-model LoRA validation or a full-matrix speed claim.', '',
    '### Actual main/isolated-repair images', '',
    'These are complete generated source images at 1:1 scale, with labels outside the image bounds. The repair column is not the unchanged merged PR head. Three high prompts and one returned-lossless example are shown per shape. All 12 panels were visually inspected separately from numerical checks.',
]
for row in manifest['rows']:
    name = Path(row['panel']).name
    assert digest(artifacts/'media'/name) == row['panel_sha256']
    body += ['', f'![Actual {row["shape"]} {row["quality"]} {row["request"]}: frozen main vs isolated lifetime repair]({raw}/media/{name})']
worst = min(diagnostics['quality_rows'], key=lambda row: row['ssim'])
body += ['', f'The worst audited SSIM request is {worst["shape"]}/{worst["quality"]}/{worst["request"]}: {worst["ssim"]:.6f}. Repeated-short results are retained in the full numerical audit.', '',
    '### Reproduction and retained failure evidence', '',
    '; '.join([
        link('Evidence and reproduction', 'README.md'),
        link('All measurements', 'performance-table.md'),
        link('Actual regression receipt', 'evidence/preflight/joy-input-lifetime-repair1-tests-admission.json'),
        link('Diagnostic audit', 'evidence/preflight/joy-input-lifetime-repair1-diagnostics1-admission.json'),
        link('Unchanged failed official-head admission', 'evidence/preflight/joy-current-head-square-primary1-admission.json'),
        link('Media provenance', 'media/manifest.json'),
        link('Visual review', 'media/visual-review.json'),
    ])+'.', '',
    'All four A/A/primary archives and the diagnostic archive were verified before export. The public package includes process measurements, commands, telemetry, source identities, quality records and archive member hashes. Full original archives remain local; public PNGs are deduplicated by hash and repeated base64 responses retain their original hashes. No weights, tensor snapshots or Git transfer bundles enter the code change.', '',
    'The repair is a separate local source revision. Publishing these measurements does not submit a replacement optimization PR, amend the merged PR head, or resolve the qualification/counting decision.',
    '<!-- /joy-isolated-input-lifetime-repair-20261030 -->', '',
]
output = evidence/'joy-input-lifetime-repair1-pr47022-supplement1.md'
with output.open('x') as stream:
    stream.write('\n'.join(body))
print(json.dumps(dict(supplement=str(output), artifact_commit=args.artifact_commit, source_sha=head,
    embedded_panels=12, official_PR_head_qualified=False, publication_pending=True,
    requirement='Refresh actual PR head/body; preserve all existing failure/history/bot content and verify rendered new images after any authorized publication.')))
