"""Record final audited results without changing frozen workload contracts."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'
now = datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')

precision = dict(status='audited', updated_at=now, primary_timing=False,
    source_head='efcb88a201a41b1fecc516a60df0ca0a34fb6dd5',
    checkpoint_revision='d26bb61231c349cf6b7896fa83353113880e1ba3',
    lane='BF16 native components, native mixed-precision internal arithmetic; no quantization',
    original_stage_evidence='qwen21-native-stage-tensor-comparison.json',
    equivalent_boundary_evidence='qwen21-native-stage-equivalent-boundaries.json',
    sigma_schedule_evidence='qwen21-scheduler-rounding-attribution-v2.json',
    limits=[
        'Original intermediate captures predate final candidate; final output/dispatch audits separately validate the final head.',
        'Effective BF16 model timesteps match at 32 recorded boundaries (4 steps per request); this is not an exhaustive 50-step equality claim.',
        'The at-most-two-FP32-ULP discrepancy describes the sigma schedule, not all solver arithmetic.',
        'T2I text hidden states match. Edit pre-collapse hidden states differ; cross-framework intermediate equivalence is not claimed.',
        'Final native Omni stage observer records metadata only and is excluded from primary timing.'
    ], final_omni_observations=[])
for folder in sorted((E/'diagnostics').glob('qwen21-gate-up-v4-final-*-diagnostics1-*-omni_*')):
    files=list(folder.glob('*.jsonl'))
    assert len(files)==1
    rows=[json.loads(line) for line in files[0].read_text().splitlines()]
    actual=[r for r in rows if len(r['scheduler_calls'])==50]
    startup=[r for r in rows if len(r['scheduler_calls'])!=50]
    assert actual and len(startup)==1 and len(startup[0]['scheduler_calls'])==2
    for r in actual:
        assert not r['cuda_autocast']
        for call in r['scheduler_calls']:
            assert not call['cuda_autocast']
            assert all(call[k]['dtype']=='torch.bfloat16' for k in ['model_output','sample','result'])
    precision['final_omni_observations'].append(dict(file=str(files[0].relative_to(E)),
        actual_requests=len(actual), scheduler_calls=50*len(actual),
        scheduler_boundary_dtypes='BF16 model_output/sample/result', cuda_autocast=False,
        excluded_startup_requests=len(startup)))
write(E/'preflight/qwen21-final-runtime-precision-audit.json', precision)

summary=read(E/'preflight/qwen21-gate-up-v4-final-results-summary.json')
execution=dict(status='final_validation_complete_publication_pending', updated_at=now,
    head=summary['head'], active_runner_pid=None,
    results='evidence/preflight/qwen21-gate-up-v4-final-results-summary.json',
    cases={k:dict(paired_comparisons=v['paired_comparisons'],
        cross_framework_minimum_metrics=v['cross_framework_minimum_metrics'],
        all_sgl_RGBA_exact_main=v['main_final_RGBA_byte_exact']) for k,v in summary['cases'].items()},
    caveats=['Original failed cleanup screens and Omni CV screens retained; see separate explicit admission records.',
             '1024 main-relative benefit is below 1.5%; the 512 cases qualify the performance claim.',
             'BCG compatibility only, no net performance win claimed.',
             'Single-GPU batch/concurrency one; broader frozen coverage remains incomplete.'])
matrix=read(ROOT/'matrix-draft.json')
cell=next(c for c in matrix['cells'] if c['id']=='QwenImage21Pipeline')
cell.setdefault('execution_history',[]).append(cell.get('current_execution',{}))
cell['current_execution']=execution
for row in cell['workloads']:
    if row['id'] in ['qwen21-base-t2i','qwen21-base-edit']:
        row['final_high_execution']=dict(execution, case_id=row['task']+'1024')
for row in matrix['supplemental_cells']:
    if row['id'] in ['qwen21-base-t2i-512','qwen21-base-edit-512']:
        row['execution']=dict(execution, case_id=row['task']+'512')
write(ROOT/'matrix-draft.json',matrix)
ledger=read(ROOT/'pr-ledger.json')
assert ledger['qualified_pr_count']==2
q=ledger['qwen21_execution']
q['phase']=execution['status']; q['active_runner_pid']=None; q['active_run']=None
q['resume4_active']=dict(status='terminal', completed=True,
    note='All four final runs, raw audits, profiles and media preparation completed. Prior PID fields were historical.')
q['final_validation']=execution
write(ROOT/'pr-ledger.json',ledger)
(ROOT/'STATUS.md').write_text('# Active: 2/10 qualified Draft PRs\n\nQwen final head '+summary['head']+
    ' has completed all final primary, quality, native-stage and profile audits. GPU jobs are terminal. '+
    'Public evidence and the official Draft PR are being prepared; Qwen is not counted yet. '+
    'Original noise and cleanup screen failures remain disclosed in admission records.\n')
with (ROOT/'CONTINUE.md').open('a') as f:
    f.write('\n## Final validation complete — '+now+'\n\n'+
      'All four final queue runs are terminal; 1024 native-stage raw audit passes. '+
      'Final summary: evidence/preflight/qwen21-gate-up-v4-final-results-summary.json. '+
      'Main/PR paired latency reductions T2I/edit: 512 3.16193%/2.63035%; 1024 1.08474%/0.93635%. '+
      'Omni/PR: 512 8.69682%/27.96034%; 1024 11.70646%/12.01035%. '+
      'All final SGL outputs exact main; all selected Omni RGB/alpha gates pass. '+
      'Omni CV failures remain explicit (512 edit .621%, 1024 T2I .553%/1.044%); separate post-screen admission is disclosed. '+
      'Both final 1024 profiles verified 416→384 GPU kernels, 64→32 gate/up GEMMs. '+
      'All four main/PR panels inspected locally. Still required: publish evidence and official Draft, verify actual page media, then count PR3. '+
      'Two GPUs idle verified after all jobs; no downloads active. Earlier active-PID notes above are historical.\n')
print('Final precision and bookkeeping saved; qualified count remains 2.')
