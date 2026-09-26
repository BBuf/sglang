#!/usr/bin/env python3
"""Audit every native A/B saved output against the same-source first baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument('--abba',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--require-exact',action='store_true',help='Require byte/pixel/audio exactness even under an existing high model policy.')
a=p.parse_args()
root=Path(__file__).resolve().parent
measurements=json.loads(a.abba.read_text())
rows=measurements['rows'];assert len(rows)==8
reference=next(row for row in rows if row['arm']=='baseline')
reference_path=Path(reference['artifacts'][0]['path'])
a.output.mkdir(parents=True,exist_ok=False)
comparisons=[]
source_manifests={}
environments=[]
for row in rows:
 assert row['valid'] and not row['diagnostic']
 assert row['request']==reference['request'] and row['quality']==reference['quality']
 assert row.get('deterministic_audio_decode',False)==reference.get('deterministic_audio_decode',False)
 assert row['gpu']==reference['gpu']
 row_dir=Path(row['artifacts'][0]['path']).parent
 manifest=json.loads((row_dir/'source-files.json').read_text())
 if row['arm'] in source_manifests:
  assert source_manifests[row['arm']]==manifest, f'Source files changed within {row["arm"]} repeats'
 else:source_manifests[row['arm']]=manifest
 environment=json.loads((row_dir/'environment.json').read_text())
 if environments:assert environment==environments[0], 'Runtime versions changed between measurements'
 environments.append(environment)
 output=a.output/row['label']
 path=Path(row['artifacts'][0]['path'])
 # Same-source repeats are held to exactness even for an existing high policy.
 gate='lossless' if row['arm']=='baseline' or a.require_exact else row['quality']
 subprocess.run([sys.executable,str(root/'compare_media.py'),'--before',str(reference_path),'--after',str(path),'--quality',gate,'--output',str(output)],check=True)
 result=json.loads((output/'comparison.json').read_text())
 comparisons.append(dict(label=row['label'],arm=row['arm'],quality=row['quality'],comparison=result))
changed_source_files=[path for path in source_manifests['baseline'].keys()|source_manifests['candidate'].keys() if source_manifests['baseline'].get(path)!=source_manifests['candidate'].get(path)]
summary=dict(qualified=True,scope='full native saved request; all eight ABBA rows compared',request=reference['request'],baseline_sha=reference['source_sha'],candidate_sha=next(row['source_sha'] for row in rows if row['arm']=='candidate'),quality=reference['quality'],comparisons=comparisons,timing_groups=measurements['groups'],model_speedup_qualified=measurements['model_speedup_qualified'],deterministic_audio_decode=reference.get('deterministic_audio_decode',False),changed_source_files=sorted(changed_source_files),source_manifests_stable=True,runtime_versions_stable=True)
(a.output/'evidence.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({key:value for key,value in summary.items() if key!='comparisons'},indent=2))
