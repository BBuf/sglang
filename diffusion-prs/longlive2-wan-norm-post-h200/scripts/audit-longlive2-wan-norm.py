"""Audit completed LongLive 2 native requests and decoded outputs; keep weights until reviewed."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root=Path('/campaign');art=root/'artifacts/longlive2'
done=root/'validate-longlive2-wan-norm.exit'
while not done.exists():time.sleep(5)
assert done.read_text().strip()=='0',done
extra=root/'repeat-longlive2-wan-norm-final.exit'
while not extra.exists():time.sleep(5)
assert extra.read_text().strip()=='0',extra
head=(art/'t2v-norm-r2-b1'/'source.txt').read_text().splitlines()[0]
groups={};profiles={};outputs=[]
for mode in ('t2v','i2v'):
    for repeat in (1,2,3,4):
        rows=[]
        for arm in ('a1','b1','b2','a2'):
            label=f'{mode}-norm-r{repeat}-{arm}'
            result=json.loads((art/label/'result.json').read_text())
            assert not result['error'] and not result['breakable_cuda_graph']
            source=(art/label/'source.txt').read_text().splitlines()[0]
            assert source==('993d1fccbaafe3e79d91567d2fc1d665cc94fa50' if arm.startswith('a') else (head if repeat<=2 else (root/'artifacts/longlive2/repeat-source.txt').read_text().strip()))
            log=re.sub(r'\x1b\[[0-9;]*m','',(root/f'longlive2-{label}.log').read_text())
            client=re.findall(r'Pixel data generated successfully in ([\d.]+) seconds',log)
            assert len(client)==1
            rows.append(dict(label=label,arm=arm,source=source,worker_s=result['e2e_latency_s'],client_s=float(client[0]),denoise_s=result['denoise_latency_s'],peak_gib=result['peak_memory_gb']))
        means={}
        for key in ('worker_s','client_s','denoise_s'):
            a=statistics.mean(r[key] for r in rows if r['arm'].startswith('a'))
            b=statistics.mean(r[key] for r in rows if r['arm'].startswith('b'))
            means[key]=dict(baseline=a,candidate=b,reduction_pct=(a-b)/a*100)
        groups[f'{mode}-r{repeat}']=dict(rows=rows,means=means,qualified=all(means[k]['reduction_pct']>=1.5 for k in ('worker_s','client_s')))
    base_profile=f'{mode}-baseline-profile'+('-retry1' if mode=='t2v' else '')
    for label in (base_profile,f'{mode}-norm-profile',f'{mode}-norm-final-profile'):
        trace=next((art/label/'traces').glob('*full_stages*.trace.json.gz'))
        with (trace.parent/'wan-norm-analysis.txt').open('w') as out:
            subprocess.run(['python',str(root/'analyze-longlive2-norm-trace.py'),str(trace)],stdout=out,check=True)
        report=json.loads(trace.with_name('wan-norm-silu-evidence.json').read_text())
        assert report['scope_count']==464
        fused=[k for k in report['kernels'] if 'wan_norm_silu_post_kernel' in k['name']]
        assert (0 < sum(k['count'] for k in fused) <= 464) if 'norm-' in label else not fused,(label,fused)
        report['fused_kernel_count']=sum(k['count'] for k in fused)
        report['fused_gpu_ms']=sum(k['gpu_ms'] for k in fused)
        profiles[label]=report
    # Native I2V preserves the 3:2 source aspect ratio within the requested pixel budget.
    # ffprobe confirms 1152x768 for all 24 baseline/candidate I2V artifacts.
    width,height=(832,480) if mode=='t2v' else (1152,768)
    comparison=art/'output-comparison'/mode;comparison.mkdir(parents=True,exist_ok=True)
    ref=next((art/f'{mode}-norm-r4-a1').glob('*.mp4'))
    ref_sha=hashlib.sha256(ref.read_bytes()).hexdigest()
    def framehash(path):
        text=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(path),'-map','0:v:0','-an','-f','framehash','-hash','sha256','-'],text=True)
        hashes=[l.split(',')[-1].strip() for l in text.splitlines() if l and not l.startswith('#')]
        assert len(hashes)==61,(path,len(hashes))
        return text,hashes
    _,ref_hashes=framehash(ref);ref_frames=None
    for path in sorted(art.glob(mode+'-*/result.json')):
        r=json.loads(path.read_text());valid=not r['error']
        row=dict(label=path.parent.name,mode=mode,quality=r['quality'],valid_request=valid,valid_performance_row=valid and 'profile' not in path.parent.name,bcg=r['breakable_cuda_graph'])
        files=r.get('output_artifacts',[])
        if not valid:
            assert row['bcg'] and not files,row
            row.update(artifact_produced=False,reason='BCG explicitly disabled by the native LongLive 2 config')
            outputs.append(row);continue
        assert len(files)==1
        video=Path(files[0]);sha=hashlib.sha256(video.read_bytes()).hexdigest();assert r['output_sha256']==[sha]
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))['streams']
        v=next(s for s in streams if s['codec_type']=='video')
        assert not any(s['codec_type']=='audio' for s in streams)
        assert (v['width'],v['height'],int(v['nb_frames']),v['r_frame_rate'])==(width,height,61,'24/1')
        text,hashes=framehash(video);(comparison/f'{path.parent.name}.framehash').write_text(text)
        exact=hashes==ref_hashes
        row.update(artifact_produced=True,path=str(video),sha256=sha,byte_exact=sha==ref_sha,pixel_exact=exact,frame_count=61,video_stream=v,audio_streams=0)
        if r['quality']=='lossless':assert exact and sha==ref_sha,row
        elif not exact:
            def decode(p):
                return np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),'-map','0:v:0','-an','-f','rawvideo','-pix_fmt','rgb24','-']),dtype=np.uint8).reshape(61,height,width,3)
            if ref_frames is None:ref_frames=decode(ref)
            frames=decode(video)
            ssims=[float(structural_similarity(a,b,channel_axis=-1,data_range=255)) for a,b in zip(ref_frames,frames)]
            psnrs=[float(peak_signal_noise_ratio(a,b,data_range=255)) for a,b in zip(ref_frames,frames)]
            row.update(ssim_min=min(ssims),ssim_mean=statistics.mean(ssims),psnr_min_db=min(psnrs),psnr_mean_db=statistics.mean(psnrs))
            row['quality_pass']=min(ssims)>=.92 and min(psnrs)>=24
            del frames
        else:row['quality_pass']=True
        outputs.append(row)
    high=[r for r in outputs if r['mode']==mode and r['quality']=='high' and r['valid_request']]
    assert len(high)==3
    assert len({r['sha256'] for r in high})==1,'high baseline/candidate output changed'
    strips=[]
    for label,title in [(f'{mode}-norm-r4-a1','Baseline eager'),(f'{mode}-norm-r4-b1','Candidate eager')]:
        video=next((art/label).glob('*.mp4'));strip=comparison/f'{label}-strip.png'
        subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(video),'-vf',"select='eq(n,0)+eq(n,20)+eq(n,40)+eq(n,60)',scale=416:-1,tile=4x1",'-frames:v','1',str(strip)],check=True)
        strips.append((title,Image.open(strip).convert('RGB')))
    w,h=strips[0][1].size;figure=Image.new('RGB',(w,2*(h+32)),'white');draw=ImageDraw.Draw(figure)
    for i,(title,strip) in enumerate(strips):
        y=i*(h+32);figure.paste(strip,(0,y+32));draw.text((12,y+10),title+' | frames 0,20,40,60 | 61 frames, 24 fps',fill='black')
    figure.save(comparison/'comparison.jpg',quality=94)
    del ref_frames
report=dict(candidate=(art/'repeat-source.txt').read_text().strip(),previous_candidate=head,groups=groups,profiles=profiles,outputs=outputs,qualified=all(groups[f'{mode}-r{r}']['qualified'] for mode in ('t2v','i2v') for r in (3,4)),note='Profile runs excluded from E2E performance. Actual BCG disabled in both configs. Initial probe latencies excluded from repeated A/B. All four fixed ABBA groups are reported, including the first T2V candidate client outlier; the final source must qualify in both r3/r4 groups for each mode.')
(art/'final-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('profiles','outputs')},indent=2))
