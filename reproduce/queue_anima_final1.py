"""Continue only after pinned A/A and pre-cleanup ABBA admit the optimization."""
import datetime,json,subprocess,time
from pathlib import Path
R=Path('/data/xyzhang/diffusion-rtx6000-20261026')
STATUS=R/'queue-anima-final1-status.json'
assert not STATUS.exists()
def save(status,**extra):STATUS.write_text(json.dumps(dict(status=status,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
def execute(args):
 print(json.dumps(dict(command=args)),flush=True)
 subprocess.run(args,check=True)
def docker(args):execute(['docker','exec','bbuf-diffusion-rtx6000-20261026-sgl']+args)
try:
 save('waiting_for_precleanup_abba')
 while True:
  s=json.loads((R/'anima-rope-v1-cudnn-abba1-status.json').read_text())
  if s['status']!='running':break
  time.sleep(10)
 assert s['status']=='raw_complete_audit_pending',s
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 save('auditing_precleanup_abba')
 docker(['python3','/task/audit_anima_blocks1.py','/task/anima-rope-v1-cudnn-abba1.json','--output','/task/evidence/preflight/anima-rope-v1-cudnn-abba1-audit.json'])
 audit=json.loads((R/'evidence/preflight/anima-rope-v1-cudnn-abba1-audit.json').read_text())
 assert audit['all_raw_source_inputs_outputs_and_gpu_ownership_pass'] and audit['all_aa_noise_pass']
 aa=json.loads((R/'evidence/preflight/anima-main-cudnn-aa1-audit.json').read_text())
 assert aa['all_aa_noise_pass']
 noise=max(max(x['cv_percent'],x['half_median_difference_percent']) for x in aa['aa_noise'])/100
 for case in audit['paired'].values():
  comparison=case['main_over_candidate']
  assert comparison['all_pairs_faster'] and min(comparison['ratios'])>1+2*noise
  assert comparison['min_baseline_over_max_candidate']>1+noise
 save('final_head_existing_tests')
 execute(['python3',str(R/'run_anima_rope_final_tests1.py')])
 save('final_primary_comparison')
 execute(['python3',str(R/'run_anima_blocks2.py'),str(R/'anima-rope-final-compare1.json')])
 save('final_primary_raw_audit')
 docker(['python3','/task/audit_anima_blocks1.py','/task/anima-rope-final-compare1.json','--output','/task/evidence/preflight/anima-rope-final-compare1-audit.json'])
 final=json.loads((R/'evidence/preflight/anima-rope-final-compare1-audit.json').read_text())
 assert final['all_raw_source_inputs_outputs_and_gpu_ownership_pass']
 # Retain all cross-framework losses; no assertion that every comparator is beaten.
 save('final_synchronized_diagnostics')
 execute(['python3',str(R/'run_anima_blocks3.py'),str(R/'anima-rope-final-diagnostics1.json')])
 docker(['python3','/task/audit_anima_blocks1.py','/task/anima-rope-final-diagnostics1.json','--output','/task/evidence/preflight/anima-rope-final-diagnostics1-audit.json'])
 save('raw_complete_final_admission_media_and_publication_pending')
except Exception as exc:
 save('failed_preserve_all_no_retry',error=repr(exc))
 raise
