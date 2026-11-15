"""Common native async lifecycle; preserve raw requests, artifacts and timer bounds."""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from build_nomas_video_payload import build_payload

arm, run = sys.argv[1:]
assert arm in ('sgl', 'omni') and Path(run).name == run
folder = Path('/task/runs') / run
visit = json.loads((folder / 'visit.json').read_text())
contract = json.loads(Path('/task/inputs/nomas-native-workloads01.json').read_text())
out = folder / 'outputs'
out.mkdir(exist_ok=False)
session = requests.Session()
session.trust_env = False
base = 'http://127.0.0.1:' + ('32150' if arm == 'sgl' else '32151')
rows = []
try:
    for index, item in enumerate(visit['requests']):
        prompt_id = item['prompt_id']
        directory = out / f'{index:02d}-{prompt_id}'
        directory.mkdir()
        payload = build_payload(arm, contract, item, directory)
        assert not payload.get('profile', False)
        (directory / 'request.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
        server_log_start = (folder / "server.log").stat().st_size
        started_wall_ns = time.time_ns()
        started = time.perf_counter_ns()
        response = session.post(base+'/v1/videos', **({'json':payload} if arm=='sgl' else {'data':payload}), timeout=120)
        submitted = time.perf_counter_ns()
        (directory / 'submit-response.txt').write_text(response.text)
        response.raise_for_status()
        job = response.json()
        video_id = job['id']
        last_incomplete_send = started
        complete_upper = submitted
        with (directory / 'poll.jsonl').open('x') as log:
            while job['status'] not in ('completed','failed','cancelled'):
                assert time.perf_counter_ns()-started < 7200*10**9, 'generation deadline'
                time.sleep(visit['poll_seconds'])
                sent = time.perf_counter_ns()
                response = session.get(base+'/v1/videos/'+video_id,timeout=120)
                received = time.perf_counter_ns()
                response.raise_for_status()
                job = response.json()
                log.write(json.dumps(dict(send_ns=sent,receive_ns=received,job=job))+'\n')
                log.flush()
                if job['status']=='completed':
                    complete_upper=received
                else:
                    last_incomplete_send=sent
        (directory / 'terminal-response.json').write_text(json.dumps(job,indent=2)+'\n')
        assert job['status']=='completed',job
        download_start=time.perf_counter_ns()
        response=session.get(base+'/v1/videos/'+video_id+'/content',timeout=120)
        downloaded=time.perf_counter_ns()
        response.raise_for_status()
        assert 'video/mp4' in response.headers.get('Content-Type','')
        content=response.content
        (directory/'output.mp4').write_bytes(content)
        materialized=time.perf_counter_ns()
        (directory/'response-headers.json').write_text(json.dumps(dict(response.headers),indent=2)+'\n')
        with (folder / "server.log").open("rb") as log:
            log.seek(server_log_start)
            request_log = log.read().decode(errors="replace")
        (directory / "server-request.log").write_text(request_log)
        events=[]
        for p in (folder/'runtime-audit').glob('*.jsonl'):
            for line in p.read_text().splitlines():
                event=json.loads(line)
                if started_wall_ns<=event['wall_ns']<=time.time_ns():events.append(event)
        begins=[e for e in events if e['event']=='request_begin']
        predictions=[e for e in events if e['event']=='native_predict' and not e['commit']]
        commits=[e for e in events if e['event']=='native_predict' and e['commit']]
        inputs=[e for e in events if e['event']=='native_inputs']
        completed=[e for e in events if e['event']=='denoise_complete']
        assert len(begins)==len(inputs)==len(completed)==1,(begins,inputs,completed)
        assert begins[0]['seed']==item['seed'] and begins[0]['steps']==item['steps']
        assert len(predictions)==item['steps'],(len(predictions),item['steps'])
        assert inputs[0]['tensors']['initial_noise']['shape']==[1,32,(item['frames']-1)//4+1,item['height']//32,item['width']//32]
        (directory/'native-audit.json').write_text(json.dumps(events,indent=2)+'\n')
        fusion=[e for e in events if e['event']=='vae_mean_square_fusion']
        assert len(fusion)==1 and fusion[0]['verified'] and not fusion[0]['gate_disabled']
        assert sum(c['count'] for c in fusion[0]['calls'])==visit['expected_mean_square_calls']
        assert {c['shape'][-1] for c in fusion[0]['calls']}=={64,4096}
        digest=hashlib.sha256(content).hexdigest()
        expected=item.get('expected_sha256',{}).get(arm)
        row=dict(index=index,prompt_id=prompt_id,role=item['role'],purpose=visit['purpose'],
            benchmark=item['role']=='measure',profiler=False, native_steps_observed=len(predictions),native_commit_forwards=len(commits),workload_id=item["id"],seed=item["seed"],frames=item["frames"],
            start_wall_ns=started_wall_ns,
            timer='perf_counter_ns: before POST through completed GET content and local write; hash excluded; no fsync added',
            timer_ns=dict(start=started,submitted=submitted,last_incomplete_send=last_incomplete_send,
                completed_poll_receive=complete_upper,download_start=download_start,downloaded=downloaded,materialized=materialized),
            output_ready_bounds_seconds=[(last_incomplete_send-started)/1e9,(complete_upper-started)/1e9],
            saved_request_e2e_seconds=(materialized-started)/1e9,
            download_seconds=(downloaded-download_start)/1e9,client_write_seconds=(materialized-downloaded)/1e9,
            bytes=len(content),sha256=digest,expected_sha256=expected,
            matches_reference=None if expected is None else digest==expected,output=str(directory/'output.mp4'))
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        assert expected is None or digest==expected,'native output changed; preserve visit and investigate'
finally:
    session.close()
