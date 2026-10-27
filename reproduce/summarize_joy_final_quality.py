"""Summarize every cross-framework gate from the already audited final primary run."""
import argparse
import json
import math
from pathlib import Path

from analyze_joy_blocks import read

parser = argparse.ArgumentParser()
parser.add_argument('shape', choices=['square', 'wide'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
evidence = root / 'evidence'
run = f'joy-gelu-final-{args.shape}-paired1'
analysis_path = evidence / 'benchmarks' / (run + '-analysis.json')
analysis = read(analysis_path)
assert analysis['source_and_ownership_audit_pass'] and analysis['all_planned_blocks_present']
assert len(analysis['blocks']) == 6
rows = []
for block in analysis['blocks']:
    name = block['name']
    arm = 'omni' if block['config'].startswith('omni') else 'sgl'
    gate = read(evidence / 'qualification' / f'{arm}-joy-{name}-quality.json')
    assert gate['all_pass']
    for row in gate['results']:
        if row['reference'].startswith(arm + '-'):
            continue
        rows.append(dict(block=name, **{k:row[k] for k in ('prompt', 'reference', 'ssim', 'psnr_db')}, lane='qualification'))
    if arm == 'omni':
        gate = read(evidence / 'benchmarks' / name / 'quality.json')
        assert gate['all_pass']
        for row in gate['rows']:
            rows.append(dict(block=name, request_index=row['index'], warmup=row['warmup'],
                reference=gate['reference'], ssim=row['ssim'], psnr_db=row['psnr_db'], lane='warmup_or_primary'))
result = dict(run=run, source_and_output_audit=str(analysis_path.relative_to(root)),
    all_pass=True, output_count=6*(4+analysis['plan']['warmups']+analysis['plan']['repeats']),
    crossframework_comparisons=len(rows), min_ssim=min(rows, key=lambda x:x['ssim']),
    min_finite_psnr=min((r for r in rows if math.isfinite(r['psnr_db'])), key=lambda x:x['psnr_db']),
    crossframework_rows=rows,
    note='All24qualification+12warmup+24primary outputs archived/audited; SGL outputs byte-identical to declared same-source references. Omni outputs individually pass numeric gates. Actual visual reviews are recorded separately; no rerolls.')
target = evidence / 'qualification' / (run + '-quality-summary.json')
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'crossframework_rows'}, indent=2))
