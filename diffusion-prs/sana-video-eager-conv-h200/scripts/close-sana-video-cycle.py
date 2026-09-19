"""Audit native SANA Video timings, profiles and decoded output, then clean weights."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root = Path('/campaign')
art = root/'artifacts/sana-video'
head = '275113434ffe34412f807514399ba00ef70fa38e'
for name in ('validate-sana-video-conv','sana-video-quality-matrix'):
    done = root/f'{name}.exit'
    while not done.exists():
        time.sleep(5)
    assert done.read_text().strip() == '0', done
assert subprocess.check_output(['git','-C',str(root/'candidate-sana-video'),'rev-parse','HEAD'],text=True).strip() == head

groups = {}
for mode, repeats in [('eager',(1,2)), ('bcg',(1,))]:
    for repeat in repeats:
        rows = []
        for arm in ('a1','b1','b2','a2'):
            label = f'conv-{mode}-r{repeat}-{arm}'
            result = json.loads((art/label/'result.json').read_text())
            assert not result['error']
            if mode == 'bcg':
                assert result['bcg_capture_detected'] and not result.get('bcg_invalid_signals')
            source = (art/label/'source.txt').read_text().splitlines()[0]
            assert source == ('993d1fccbaafe3e79d91567d2fc1d665cc94fa50' if arm.startswith('a') else head)
            log = re.sub(r'\x1b\[[0-9;]*m','',(root/f'sana-video-{label}.log').read_text())
            client = re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
            assert len(client) == 1
            rows.append(dict(label=label,arm=arm,source=source,
                worker_s=result['e2e_latency_s'],denoise_s=result['denoise_latency_s'],
                client_s=float(client[0]),peak_gib=result['peak_memory_gb']))
        means = {}
        for key in ('worker_s','client_s','denoise_s'):
            a=statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
            b=statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
            means[key]=dict(baseline=a,candidate=b,reduction_pct=(a-b)/a*100)
        groups[f'{mode}-r{repeat}']=dict(rows=rows,means=means,
            qualified=all(means[k]['reduction_pct']>=1.5 for k in ('worker_s','client_s')))
assert all(groups[f'eager-r{r}']['qualified'] for r in (1,2)),groups

profiles = {}
for label in ('baseline-eager-profile','baseline-bcg-profile','conv-eager-profile','conv-bcg-profile'):
    trace=next((art/label/'traces').glob('*-3_steps-*.trace.json.gz'))
    with (art/label/'step-analysis.json').open('w') as out:
        subprocess.run(['python',str(root/'analyze-sana-trace.py'),str(trace)],stdout=out,check=True)
    report=json.loads(trace.with_name('denoise-step2-evidence.json').read_text())
    assert report['crossing_kernels']==0
    with gzip.open(trace.with_name('denoise-step2.trace.json.gz'),'rt') as stream:
        events=json.load(stream)['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel']
    eager_ids=set()
    for e in events:
        args=e.get('args',{});dims=args.get('Input Dims',[])
        if not dims or 'External id' not in args:
            continue
        if ((e.get('name')=='aten::add_' and dims[0]==[21,13440,30,52])
            or (e.get('name')=='aten::silu' and dims[0] in ([21,13440,30,52],[21,6720,30,52]))
            or (e.get('name')=='aten::mul' and dims[0]==[21,6720,30,52])):
            eager_ids.add(args['External id'])
    eager=[e for e in kernels if e.get('args',{}).get('External id') in eager_ids]
    fused=[e for e in kernels if e['name'] in ('_bias_silu_kernel','_bias_glu_kernel')]
    report['conv_post_attribution']=dict(eager_calls=len(eager),eager_ms=sum(e['dur'] for e in eager)/1000,
        fused_calls=len(fused),fused_ms=sum(e['dur'] for e in fused)/1000)
    assert len(fused)==(0 if label=='baseline-eager-profile' else 80),(label,len(fused))
    if 'bcg' in label:
        with gzip.open(trace,'rt') as stream:full=json.load(stream)['traceEvents']
        full_graphs=sum(e.get('name')=='cudaGraphLaunch' for e in full)
        assert full_graphs==8 and report['graph_launch_api_count']==2
        report['full_trace_graph_launches']=full_graphs
    trace.with_name('conv-post-attribution.json').write_text(json.dumps(report['conv_post_attribution'],indent=2))
    with (art/label/'triage-step2.txt').open('w') as out:
        subprocess.run(['python',str(root/'llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py'),
            '--framework','sglang','--input',str(trace.with_name('denoise-step2.trace.json.gz'))],stdout=out,check=True)
    profiles[label]=report

comparison=art/'output-comparison';comparison.mkdir(exist_ok=True)
reference=next((art/'conv-eager-r2-a1').glob('*.mp4'))
def hashes(path):
    output=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(path),
        '-map','0:v:0','-an','-f','framehash','-hash','sha256','-'],text=True)
    frames=[line.split(',')[-1].strip() for line in output.splitlines() if line and not line.startswith('#')]
    assert len(frames)==81,(path,len(frames))
    return output,frames
_,ref_hashes=hashes(reference)
ref_sha=hashlib.sha256(reference.read_bytes()).hexdigest()
ref_frames=None
outputs=[]
for path in sorted(art.glob('*/result.json')):
    result=json.loads(path.read_text());valid=not result['error'];quality=result['quality']
    if not valid:
        assert result.get('breakable_cuda_graph') and result.get('bcg_invalid_signals')
    files=result.get('output_artifacts') or [str(p) for p in path.parent.glob('*.mp4')]
    row=dict(label=path.parent.name,quality=quality,valid_performance_row=valid,
        bcg_capture_detected=result.get('bcg_capture_detected'),bcg_invalid_signals=result.get('bcg_invalid_signals',[]))
    if not files:
        assert not valid
        row['artifact_produced']=False;outputs.append(row);continue
    assert len(files)==1
    video=Path(files[0]);sha=hashlib.sha256(video.read_bytes()).hexdigest()
    if result.get('output_sha256'):assert result['output_sha256']==[sha]
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))
    streams=metadata['streams'];v=next(s for s in streams if s['codec_type']=='video')
    assert not any(s['codec_type']=='audio' for s in streams)
    assert (v['width'],v['height'],int(v['nb_frames']),v['r_frame_rate'])==(832,480,81,'16/1')
    text,frame_hashes=hashes(video);(comparison/f'{path.parent.name}.framehash').write_text(text)
    exact=frame_hashes==ref_hashes
    row.update(path=str(video),sha256=sha,byte_exact=sha==ref_sha,pixel_exact=exact,frame_count=81,video_stream=v,audio_streams=0)
    if not exact:
        assert quality=='high',row
        def decode(p):
            return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),
                '-map','0:v:0','-an','-f','rawvideo','-pix_fmt','rgb24','-']),dtype=np.uint8).reshape(81,480,832,3)
        if ref_frames is None:ref_frames=decode(reference)
        frames=decode(video)
        ssims=[float(structural_similarity(a,b,channel_axis=-1,data_range=255)) for a,b in zip(ref_frames,frames)]
        psnrs=[float(peak_signal_noise_ratio(a,b,data_range=255)) for a,b in zip(ref_frames,frames)]
        row.update(ssim_min=min(ssims),ssim_mean=statistics.mean(ssims),psnr_min_db=min(psnrs),psnr_mean_db=statistics.mean(psnrs))
        assert min(ssims)>=.92 and min(psnrs)>=24,row
    if quality=='lossless':assert exact and sha==ref_sha,row
    outputs.append(row)
assert len(outputs)==26,len(outputs)

strips=[]
for label,name in [('conv-eager-r2-a1','Baseline eager'),('conv-eager-r2-b1','Candidate eager'),('conv-bcg-r1-b1','Candidate BCG')]:
    video=next((art/label).glob('*.mp4'));strip=comparison/f'{label}-strip.png'
    subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(video),'-vf',
        "select='eq(n,0)+eq(n,26)+eq(n,53)+eq(n,80)',scale=416:-1,tile=4x1",'-frames:v','1',str(strip)],check=True)
    strips.append((name,Image.open(strip).convert('RGB')))
w,h=strips[0][1].size;figure=Image.new('RGB',(w,3*(h+32)),'white');draw=ImageDraw.Draw(figure)
for i,(name,strip) in enumerate(strips):
    y=i*(h+32);figure.paste(strip,(0,y+32));draw.text((12,y+10),name+' | frames 0,26,53,80 | 5.0625 s, 16 fps',fill='black')
figure.save(comparison/'comparison.jpg',quality=94)
final=dict(qualified=True,candidate=head,groups=groups,profiles=profiles,outputs=outputs,
    note='Eager repeated saved-request groups determine qualification. BCG is a regression comparator. Profile timings are diagnostic only.')
(art/'final-evidence.json').write_text(json.dumps(final,indent=2))

lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
selected={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in {'0','1'}}
for _ in range(30):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if not any(line.split(',')[0].strip() in selected for line in processes.splitlines()):break
    time.sleep(1)
else:raise RuntimeError('Assigned GPU busy; no process was killed')
helper=root/'baseline/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py'
spec=importlib.util.spec_from_file_location('campaign_bench',helper);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
record=bench._cleanup_model_cache(root/'model-caches',root/'model-caches/sana-video-cycle',
    root/'artifacts/cache-cleanup.jsonl','sana-video','cycle','Repeated eager ABBA, BCG comparator, paired profiles and decoded output audit complete')
assert record['after']==dict(file_count=0,weight_file_count=0,total_bytes=0)
assert not (root/'model-caches/sana-video-cycle').exists()
(art/'cache-cleanup.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(groups=groups,output_count=len(outputs),cleanup=record),indent=2))
