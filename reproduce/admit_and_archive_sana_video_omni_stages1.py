"""Audit the existing diagnostic after separating startup from HTTP requests."""
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get('DIFFUSION_TASK_ROOT', '/data/xyzhang/diffusion-rtx6000-20261026'))
RUN = 'sana-video-omni-stages1'
assert not Path('/proc/2227419').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
plan = json.loads((ROOT/(RUN+'-plan.json')).read_text())
for path, digest in plan['sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
folder = ROOT/'block-runs'/RUN
metadata = json.loads((folder/'metadata.json').read_text())
assert metadata['source_sha'] == '817d4c17feb702289a2d2b2cbacfd594c262b7a7'
assert metadata['native_synchronized_stage_profiler'] and not metadata['benchmark']
lines = (folder/'server.log').read_text().splitlines()
boundaries = [i for i,line in enumerate(lines) if 'Application startup complete.' in line]
assert len(boundaries) == 1
boundary = boundaries[0]
posts = [i for i,line in enumerate(lines) if '"POST /v1/videos/sync HTTP/1.1" 200 OK' in line]
assert len(posts) == 4
groups, stages, start = [], {}, None
for index,line in enumerate(lines):
    match = re.search(r'\[DiffusionPipelineProfiler\] SanaVideoPipeline\.([\w.]+) took ([\d.]+)s',line)
    if not match:
        continue
    name, seconds = match.groups()
    if start is None:
        start = index
    assert name not in stages
    stages[name] = float(seconds)
    if name == 'forward':
        assert set(stages) == {'encode_prompt','diffuse','vae.decode','forward'}
        groups.append(dict(start_line=start,end_line=index,stages=stages))
        stages, start = {}, None
assert not stages
warmup = [group for group in groups if group['end_line'] < boundary]
requests = [group for group in groups if group['start_line'] > boundary]
assert len(warmup) == 1 and len(requests) == 4 and len(groups) == 5
observed = ROOT/'outputs/qualification/omni-sana-video-t2v-stages1'
native = ROOT/'outputs/qualification/omni-sana-video-t2v-native1'
actual = json.loads((observed/'results.json').read_text())
reference = json.loads((native/'results.json').read_text())
assert len(actual) == len(reference) == len(requests) == 4
rows = []
for index,(prompt,group,post,measured,baseline) in enumerate(zip(['short','long','chinese','short'],requests,posts,actual,reference)):
    assert group['end_line'] < post
    if index+1 < len(requests):
        assert post < requests[index+1]['start_line']
    video = observed/f'{index:02d}-{prompt}'/'output.mp4'
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    assert digest == measured['sha256'] == baseline['sha256']
    assert measured['frame_sha256'] == baseline['frame_sha256'] and len(measured['frame_sha256']) == 81
    rows.append(dict(index=index,prompt=prompt,excluded_first_request=index==0,
                     cross_framework_quality_eligible=prompt!='chinese',
                     native_MP4_exact=True,all81frame_hashes_exact=True,video_sha256=digest,
                     http_response_log_line=post,**group))
report = dict(passed=True,benchmark=False,source_sha=metadata['source_sha'],rows=rows,
              startup_warmup=warmup,application_startup_complete_log_line=boundary,
              original_driver_status=metadata['status'],original_driver_error=metadata.get('error'),
              correction='Original collector expected four pipeline forwards; native startup also performs one forward. Preserve the original failure and all five timer groups. Separate startup by the actual Application startup complete marker and match each later group to one successful HTTP response. No GPU rerun, changed timer, changed request or primary measurement.',
              source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in [folder/'server.log',folder/'metadata.json',observed/'results.json',native/'results.json']},
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
admission = ROOT/'evidence/preflight'/(RUN+'-admission.json')
assert not admission.exists()
admission.write_text(json.dumps(report,indent=2)+'\n')
paths = {path for area in [folder,observed] for path in area.rglob('*') if path.is_file()}
paths.update(ROOT/path for path in plan['sha256'] if path != 'sana-video-stage-observer1/sitecustomize.py')
paths.update([admission,Path(__file__),ROOT/(RUN+'-status.json'),ROOT/(RUN+'-plan.json'),ROOT/(RUN+'-driver.log'),ROOT/(RUN+'-launch.json')])
members = [dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(paths)]
archive = ROOT/(RUN+'-archive.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
    for path in sorted(paths):
        stream.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
manifest = dict(archive=archive.name,bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=members)
(ROOT/(RUN+'-archive-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'admission_passed':True,'startup_groups':len(warmup),'HTTP_groups':len(rows),
                  'archive':{key:value for key,value in manifest.items() if key!='members'}},indent=2))
