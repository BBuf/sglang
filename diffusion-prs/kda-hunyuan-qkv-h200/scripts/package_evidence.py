#!/usr/bin/env python3
"""Package reviewed native evidence; media publication remains a separate action."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--baseline-dir',required=True,type=Path)
p.add_argument('--candidate-dir',required=True,type=Path)
p.add_argument('--media-dir',required=True,type=Path)
p.add_argument('--abba',required=True,type=Path)
p.add_argument('--dispatch-evidence',required=True,type=Path)
p.add_argument('--kda-summary',required=True,type=Path,help='Sanitized KDA provenance and kernel speed measurements; no secrets.')
p.add_argument('--model-evidence',required=True,type=Path,help='Successful all-eight-row audit_abba evidence.json.')
p.add_argument('--output',required=True,type=Path)
p.add_argument('--media-repo',default='BBuf/sglang')
p.add_argument('--media-commit',default='MEDIA_COMMIT_SHA',help='Replace placeholder after publishing the immutable evidence commit.')
p.add_argument('--media-path',required=True)
a=p.parse_args()
assert a.media_commit == 'MEDIA_COMMIT_SHA' or (len(a.media_commit)==40 and all(c in '0123456789abcdef' for c in a.media_commit))
meta={}
for name,path in [('baseline',a.baseline_dir),('candidate',a.candidate_dir)]:
 result=json.loads((path/'result.json').read_text())
 assert result['valid'] and not result['diagnostic']
 meta[name]=result
assert meta['baseline']['request']==meta['candidate']['request']
assert meta['baseline']['quality']==meta['candidate']['quality']
assert meta['baseline']['gpu']==meta['candidate']['gpu']
assert meta['baseline'].get('deterministic_audio_decode',False)==meta['candidate'].get('deterministic_audio_decode',False)
model_evidence=json.loads(a.model_evidence.read_text())
assert model_evidence['qualified'] and len(model_evidence['comparisons'])==8
assert model_evidence['baseline_sha']==meta['baseline']['source_sha']
assert model_evidence['candidate_sha']==meta['candidate']['source_sha']
quality=json.loads((a.media_dir/'comparison.json').read_text())
assert quality['qualified']
assert quality['before_sha256'] in [r['sha256'] for r in meta['baseline']['artifacts']]
assert quality['after_sha256'] in [r['sha256'] for r in meta['candidate']['artifacts']]
a.output.mkdir(parents=True,exist_ok=False)
for name,path in [('baseline',a.baseline_dir),('candidate',a.candidate_dir)]:
 target=a.output/'raw'/name;target.mkdir(parents=True)
 for file in ['request.json','result.json','perf.json','native.log','gpu-before.txt','source-files.json','environment.json']:
  shutil.copy2(path/file,target/file)
shutil.copytree(a.media_dir,a.output/'output-comparison')
for name,path in [('abba.json',a.abba),('dispatch-evidence.json',a.dispatch_evidence),('kda-summary.json',a.kda_summary),('all-model-comparisons.json',a.model_evidence)]:
 shutil.copy2(path,a.output/name)
for file in Path(__file__).parent.glob('*.py'):
 dest=a.output/'scripts';dest.mkdir(exist_ok=True);shutil.copy2(file,dest/file.name)
for directory in ['capture','repro']:
 shutil.copytree(Path(__file__).parent/directory,a.output/'scripts'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
raw=f'https://raw.githubusercontent.com/{a.media_repo}/{a.media_commit}/{a.media_path}'
blob=f'https://github.com/{a.media_repo}/blob/{a.media_commit}/{a.media_path}'
kind='gif' if quality['media_type']=='video' else 'png'
request=meta['baseline']['request']
body=f'''Native model: `{request['model']}@{request['revision']}`; quality `{meta['baseline']['quality']}`; fixed seed `{request['seed']}`; shape `{request['width']}x{request['height']}`, `{request['frames']}` frames, `{request['steps']}` steps. Prompt: `{request['prompt']}`.

![Before and after generated {quality['media_type']}]({raw}/output-comparison/comparison.{kind})

[Full per-frame/byte comparison]({blob}/output-comparison/comparison.json), [repeated native timings]({blob}/abba.json), [native kernel dispatch evidence]({blob}/dispatch-evidence.json), [KDA provenance and kernel measurements]({blob}/kda-summary.json).

This snippet accompanies a PR whose title begins `[KDA]`. Include the measured kernel speedup, actual model E2E change, tests and any applicability limits in the complete description.
'''
if meta['baseline'].get('deterministic_audio_decode'):
 body+='\nBoth arms use an explicitly recorded validation-only audio decode context: deterministic cuDNN, no cuDNN benchmark, and TF32 disabled for the audio VAE/vocoder. Default same-source repeats showed audio non-determinism; these timings apply to the stated reproduction configuration. Production model code, weights, prompt, quality and exact media gates are unchanged.\n'
(a.output/'pr-media-snippet.md').write_text(body)
manifest={str(file.relative_to(a.output)):hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(a.output.rglob('*')) if file.is_file()}
(a.output/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
print(a.output)
