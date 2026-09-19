"""Audit complete SenseNova evidence, then remove only its owned checkpoint cache."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

from PIL import Image

root=Path('/campaign')
art=root/'artifacts/sensenova-u1'
deadline=time.monotonic()+2400
for name in ['sensenova-rope-abba-r2','sensenova-rope-post','sensenova-committed-microbench']:
    terminal=root/(name+'.exit')
    while not terminal.exists():
        if time.monotonic()>deadline: raise TimeoutError(name)
        time.sleep(5)
    assert terminal.read_text().strip()=='0',(name,terminal.read_text())
for repo,sha in [('baseline-sensenova-profile','e8d68ce11f1bc1e880112da35606e8b1fd93db99'),
                 ('candidate-sensenova-rope','853522eff16a7294d9a4d2de21658dc260a9d2b6')]:
    assert subprocess.check_output(['git','-C',str(root/repo),'rev-parse','HEAD'],text=True).strip()==sha
    subprocess.run(['git','-C',str(root/repo),'diff','--quiet'],check=True)
    subprocess.run(['git','-C',str(root/repo),'diff','--cached','--quiet'],check=True)
rows=[]
images=[]
for group in ['r1','r2']:
    for arm in ['a1','b1','b2','a2']:
        label=f'rope-{group}-{arm}'
        directory=art/label
        result=json.loads((directory/'result.json').read_text())
        assert not result['error']
        perf=json.loads(next(directory.glob('sensenova-u1_*.json')).read_text())
        stage=[e['duration_ms']/1000 for e in perf['steps'] if e['name']=='SenseNovaU1GenerationStage']
        assert len(stage)==1
        log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'sensenova-{label}.log').read_text())
        client=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
        assert len(client)==1,(label,client)
        rows.append(dict(group=group,arm=arm,label=label,engine_e2e_s=result['e2e_latency_s'],
            generation_stage_s=stage[0],client_e2e_s=float(client[0]),client_precision_s=0.01,
            peak_gib=result['peak_memory_gb'],source=(directory/'source.txt').read_text().splitlines()[0]))
for label in [r['label'] for r in rows]+['profile-controls-profile','rope-r1-profile']:
    path=next((art/label).glob('*.png'))
    raw=path.read_bytes()
    with Image.open(path) as img:
        assert img.size==(2048,2048)
        img.load()
        images.append(dict(label=label,file=path.name,mode=img.mode,size=list(img.size),
            png_sha256=hashlib.sha256(raw).hexdigest(),decoded_sha256=hashlib.sha256(img.tobytes()).hexdigest()))
assert len({x['png_sha256'] for x in images})==1
assert len({x['decoded_sha256'] for x in images})==1
groups={}
for group in ['r1','r2']:
    metrics={}
    for metric in ['engine_e2e_s','client_e2e_s','generation_stage_s']:
        a=statistics.mean(r[metric] for r in rows if r['group']==group and r['arm'].startswith('a'))
        b=statistics.mean(r[metric] for r in rows if r['group']==group and r['arm'].startswith('b'))
        metrics[metric]=dict(baseline_mean=a,candidate_mean=b,reduction_pct=(1-b/a)*100)
    groups[group]=metrics
assert groups['r2']['engine_e2e_s']['reduction_pct']>=1.5,groups
assert groups['r2']['client_e2e_s']['reduction_pct']>=1.5,groups
profile={}
for arm,label in [('baseline','profile-controls-profile'),('candidate','rope-r1-profile')]:
    evidence=json.loads((art/label/'traces/denoise-step2-evidence.json').read_text())
    assert evidence['crossing_kernel_count']==0 and evidence['rope_calls']==252
    profile[arm]=evidence
assert profile['candidate']['rope_gpu_ms']<profile['baseline']['rope_gpu_ms']
summary=dict(rows=rows,groups=groups,primary_group='r2',
    r1_client_note='A1 has higher non-worker overhead; r2 repeats after CPU trace analysis and GPU microbench jobs complete.',
    definitions=dict(engine_e2e='Worker request duration through output conversion, before SAVE_OUTPUTS; excludes model loading, warmup, and PNG/file transport.',
        client_e2e='Native log_generation_timer around scheduler request, result processing and output save; rounded to 0.01s in original log.',
        generation_stage='SenseNovaU1GenerationStage, including prefixes, denoising and pixel denormalization; not an isolated denoise timer.'),
    image_comparison=dict(png_byte_identical=True,decoded_pixels_identical=True,max_abs_pixel_difference=0,ssim=1.0,psnr_db='infinity',images=images),
    profile=profile,bcg='Native pipeline disabled; eager fallback rejected',qualified=True,submitted_pr=None)
(art/'final-evidence.json').write_text(json.dumps(summary,indent=2))
lock=(root/'gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
assert not any(line.split(',')[0].strip() in selected for line in processes.splitlines()),processes
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper)
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/sensenova-u1-cycle',
    root/'artifacts/cache-cleanup.jsonl','sensenova-u1','cycle',
    'Qualified lossless eager RoPE optimization; complete paired profiles, two ABBA groups, microbench and exact PNG audit retained; PR publication follows')
assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0),record
assert not (root/'model-caches/sensenova-u1-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(groups=groups,profile={k:dict(rope_gpu_ms=v['rope_gpu_ms'],rope_kernel_count=v['rope_kernel_count']) for k,v in profile.items()},cleanup=record),indent=2),flush=True)
