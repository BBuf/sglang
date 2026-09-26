#!/usr/bin/env python3
"""Two independent unprofiled A-B-B-A groups; all raw rows remain retained."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True)
p.add_argument('--candidate',required=True)
p.add_argument('--baseline-sha',required=True)
p.add_argument('--candidate-sha',required=True)
p.add_argument('--model',required=True,choices=('zimage','wan21','sana-video','ltx2','flux2-klein','hunyuanvideo','ltx23','fasth3-vsa'))
p.add_argument('--quality',default='lossless',choices=('lossless','high'))
p.add_argument('--gpu',default='7')
p.add_argument('--output-root',required=True,type=Path)
p.add_argument('--label',required=True)
p.add_argument('--resume-completed',action='store_true',help='Reuse only successful immutable rows matching this exact source/model/quality/GPU. Never restart partial rows.')
p.add_argument('--deterministic-audio-decode',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parent
rows=[]
for repeat in (1,2):
 for arm in ('a1','b1','b2','a2'):
  baseline=arm.startswith('a')
  label=f'{a.label}-r{repeat}-{arm}'
  cmd=[sys.executable,str(root/'run_native.py'),'--repo',a.baseline if baseline else a.candidate,
       '--source-sha',a.baseline_sha if baseline else a.candidate_sha,'--model',a.model,
       '--quality',a.quality,'--gpu',a.gpu,'--output-root',str(a.output_root),'--label',label]
  if a.deterministic_audio_decode:cmd.append('--deterministic-audio-decode')
  result_path=a.output_root/label/'result.json'
  if a.resume_completed and result_path.is_file():
   row=json.loads(result_path.read_text())
   assert row['source_sha']==(a.baseline_sha if baseline else a.candidate_sha)
   assert row['repo']==str(Path(a.baseline if baseline else a.candidate).resolve())
   assert row['quality']==a.quality and row['gpu']==a.gpu
   assert row.get('deterministic_audio_decode',False)==a.deterministic_audio_decode
   assert row['request']==json.loads((root/'presets.json').read_text())[a.model]
  else:
   subprocess.run(cmd,check=True)
   row=json.loads(result_path.read_text())
  assert row['valid'] and not row['diagnostic']
  row.update(arm='baseline' if baseline else 'candidate',repeat=repeat,label=label)
  rows.append(row)
  (a.output_root/(a.label+'-abba-progress.json')).write_text(json.dumps(rows,indent=2))
groups={}
for repeat in (1,2):
 means={}
 for metric in ('client_saved_e2e_s','worker_e2e_s','denoise_s','decode_s'):
  by_arm={arm:statistics.mean(row[metric] for row in rows if row['repeat']==repeat and row['arm']==arm) for arm in ('baseline','candidate')}
  by_arm['reduction_pct']=100*(by_arm['baseline']-by_arm['candidate'])/by_arm['baseline'] if by_arm['baseline'] else None
  means[metric]=by_arm
 groups[str(repeat)]=means
summary=dict(rows=rows,groups=groups,deterministic_audio_decode=a.deterministic_audio_decode,model_speedup_qualified=all(group['client_saved_e2e_s']['reduction_pct']>=1.5 for group in groups.values()),
             note='Client saved-request E2E comes from the native Pixel data timer including scheduler wait and output save. Source verified at diffusion_generator.py:329. Model media, numerical checks, and dispatch evidence remain separate gates.')
(a.output_root/(a.label+'-abba.json')).write_text(json.dumps(summary,indent=2))
print(json.dumps(groups,indent=2))
