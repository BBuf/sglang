"""Archive audit for native final-head quality transitions and observed BCG lifecycle."""
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = 'qwen21-gate-up-v4-compatibility1'
records, manifests, warmups = [], [], []
for config in ['main_eager','final_eager','final_bcg']:
    name = RUN+'-'+config
    block = ROOT/'evidence/block-runs'/name
    meta = json.loads((block/'metadata.json').read_text())
    assert meta['status']=='complete'
    assert meta['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
    observed = [json.loads(line) for path in (ROOT/'evidence/diagnostics'/name).glob('*.jsonl') for line in path.read_text().splitlines()]
    by_id = {row['request_id']:row for row in observed if not row['is_warmup']}
    warmups.extend(dict(config=config, **row) for row in observed if row['is_warmup'])
    for entry in meta['requests']:
        label = f'base-{entry["case"]}-{RUN}-{config}-{entry["phase"]}-{entry["quality"]}'
        folder = ROOT/'evidence/qualification'/('sgl-qwen21-'+label)
        qm = json.loads((folder/'metadata.json').read_text())
        assert qm['source_sha']==meta['source_sha'] and qm['source_diff_sha256']==meta['source_diff_sha256']
        contract = json.loads((folder/'contract.json').read_text())
        assert hashlib.sha256((folder/'contract.json').read_bytes()).hexdigest()==entry['contract_sha256']
        rows = json.loads((folder/'results.json').read_text())
        references = json.loads((ROOT/'evidence/qualification'/entry['reference']/'results.json').read_text())
        assert len(rows)==entry['output_count'] and len(references)>=len(rows)
        for row,reference in zip(rows,references):
            directory = folder/Path(row['request']).name
            payload = json.loads((directory/'request.json').read_text())['payload']
            assert payload['quality']==entry['quality']
            assert int(payload['seed'])==1101 and int(payload['num_inference_steps'])==50 and float(payload['true_cfg_scale'])==1
            assert payload['size']==f'{contract["width"]}x{contract["height"]}'
            png = (directory/'output.png').read_bytes()
            assert hashlib.sha256(png).hexdigest()==row['sha256']==reference['sha256']
            assert base64.b64decode(json.loads((directory/'response.json').read_text())['data'][0]['b64_json'],validate=True)==png
            with Image.open(directory/'output.png') as image:
                assert image.mode=='RGBA' and image.size==(contract['width'],contract['height'])
            perf = json.loads((directory/'perf.json').read_text())
            counter = by_id.pop(perf['request_id'])
            assert counter['quality']==counter['mounted_quality']==entry['quality']
            assert counter['capture_calls']==0 and counter['hook_sha256']==meta['observer_sha256']
            replays = 49 if config=='final_bcg' and entry['quality']=='high' and entry['case'].startswith('t2i') else 0
            assert counter['replay_calls']==replays and counter['python_model_forwards']==50-replays
            packed = (50-replays)*32 if config!='main_eager' and entry['quality']=='high' else 0
            assert sum(counter['packed_python_calls'].values())==packed
            records.append(dict(config=config,case=entry['case'],phase=entry['phase'],quality=entry['quality'],
                                request_id=perf['request_id'],sha256=row['sha256'],replays=replays,
                                packed_python_calls=packed,profiled=row['profiled'],RGBA_byte_exact=True))
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                manifests.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert not by_id
assert len(records)==68
dest = ROOT/'evidence/qualification/qwen21-gate-up-v4-compatibility1-raw-audit.json'
assert not dest.exists()
dest.write_text(json.dumps(dict(status='pass',raw_request_count=len(records),profiled_count=sum(r['profiled'] for r in records),
                                no_serving_capture=True,all_RGBA_byte_exact_corresponding_main=True,
                                quality_transitions_pass=True,records=records,warmups=warmups,manifest=manifests,
                                caveat='Externalmetadataobserver and synchronized/profile runs; alltimingexcludedprimary. BCGT2Ireplayvalid,editfallbackretained,noBCGnetbenefitclaim.'),indent=2)+'\n')
print(json.dumps(dict(requests=len(records),profiled=sum(r['profiled'] for r in records),actual_replay_requests=sum(r['replays']>0 for r in records),all_RGBA_exact=True)))
