"""Four fixed native requests with synchronized stage timers, separate from E2E."""
import hashlib
import json
import re
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/xyzhang/diffusion-rtx6000-20261026')
RUN = 'sana-video-omni-stages1'
STATUS = ROOT/(RUN+'-status.json')
assert not STATUS.exists()
plan = json.loads((ROOT/(RUN+'-plan.json')).read_text())
for relative, digest in plan['sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
assert json.loads((ROOT/'evidence/preflight/sana-video-ln-final-primary-cleanup1-admission.json').read_text())['passed']
assert (ROOT/'sana-video-ln-final-primary1-archive-manifest.json').exists()
assert not Path('/proc/1901755').exists() and not Path('/proc/2174372').exists()


def idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()


assert idle()
folder = ROOT/'block-runs'/RUN
folder.mkdir(exist_ok=False)
container = 'bbuf-diffusion-rtx6000-20261026-omni'
launch = ['docker','exec',container,'sh','-c',f'echo $$ > /task/block-runs/{RUN}/server.pid; exec sh /task/launch_omni_sana_video_stages1.sh']
client = ['docker','exec',container,'python3','/task/submit_sana_video_qualification.py','omni',
          '--label','t2v-stages1','--contract','/task/inputs/sana-video-480p-t2v-workloads.json']
state = dict(status='starting',benchmark=False,native_synchronized_stage_profiler=True,
             torch_trace_profiler=False,source_sha='817d4c17feb702289a2d2b2cbacfd594c262b7a7',
             launch_command=launch,client_command=client,plan=plan)


def save():
    STATUS.write_text(json.dumps(state,indent=2)+'\n')
    (folder/'metadata.json').write_text(json.dumps(state,indent=2)+'\n')


save()
started = time.monotonic()
log = folder/'server.log'
with log.open('xb') as stream:
    server = subprocess.Popen(launch,stdout=stream,stderr=subprocess.STDOUT)
try:
    while True:
        assert server.poll() is None, 'Server exited before readiness'
        assert time.monotonic()-started < 1800
        try:
            with urllib.request.urlopen('http://127.0.0.1:31051/health',timeout=2) as response:
                ready = response.status == 200
        except Exception:
            ready = False
        if ready:
            break
        time.sleep(2)
    assert '[SanaVideoStageAudit] native synchronized timers:' in log.read_text()
    state.update(status='generating_diagnostic_outputs',launch_to_ready_seconds=time.monotonic()-started)
    save()
    with (folder/'client.log').open('xb') as stream:
        subprocess.run(client,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
    measured = ROOT/'outputs/qualification/omni-sana-video-t2v-stages1'
    native = ROOT/'outputs/qualification/omni-sana-video-t2v-native1'
    videos = sorted(measured.glob('*/output.mp4'))
    assert len(videos) == 4
    for video in videos:
        assert video.read_bytes() == (native/video.relative_to(measured)).read_bytes()
    matches = re.findall(r'\[DiffusionPipelineProfiler\] SanaVideoPipeline\.([\w.]+) took ([\d.]+)s',log.read_text())
    requests, row = [], {}
    for name, seconds in matches:
        assert name not in row, (name,row)
        row[name] = float(seconds)
        if name == 'forward':
            assert set(row) == {'encode_prompt','diffuse','vae.decode','forward'}, row
            requests.append(row)
            row = {}
    assert not row and len(requests) == 4, matches
    for index,(prompt,stages) in enumerate(zip(['short','long','chinese','short'],requests)):
        stages.update(index=index,prompt=prompt,excluded_first_request=index==0,
                      native_MP4_exact=True,cross_framework_quality_eligible=prompt!='chinese')
    state.update(status='complete_outputs_exact',requests=requests,all_four_MP4s_exact_native=True,
                 note='Separate native synchronized function-boundary timers; first short excluded from representative steady diagnostics. Chinese remains cross-framework quality-ineligible. Client profiled=false denotes no Chrome/Torch trace, not absence of these native timers. No primary E2E samples replaced.')
except Exception as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    if server.poll() is None and (folder/'server.pid').exists():
        subprocess.run(['docker','exec',container,'kill','-TERM',(folder/'server.pid').read_text().strip()],check=True)
    server.wait(timeout=60)
    for _ in range(30):
        if idle():
            break
        time.sleep(1)
    assert idle()
    save()

paths = {path for area in [folder,measured] for path in area.rglob('*') if path.is_file()}
paths.update(ROOT/path for path in plan['sha256'] if path != 'sana-video-stage-observer1/sitecustomize.py')
paths.update([STATUS,ROOT/(RUN+'-plan.json')])
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
print(json.dumps({key:value for key,value in state.items() if key not in ['plan','launch_command','client_command']},indent=2),flush=True)
