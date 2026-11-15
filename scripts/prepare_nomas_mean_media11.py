"""Render actual final-head/main videos and verify all decoded RGB frames."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
high = len(sys.argv)>1 and sys.argv[1]=='high'
frames = 49 if high else int(sys.argv[1]) if len(sys.argv)>1 else 49
assert frames in (49,121)
out = root/f'evidence/quality/nomas-{frames}-{"high-" if high else ""}main-cleanup11'
out.mkdir(parents=True,exist_ok=False)
names = (['nomas-49-high-main-scope15','nomas-49-high-PR-scope16'] if high else
         ['nomas-49-sgl-native02','nomas-49-sgl-mean-quality11'] if frames==49
         else ['nomas-121-sgl-native09','nomas-121-sgl-mean-quality13'])
runs = [root/'runs'/name for name in names]
results = [json.loads((p/'outputs/results.json').read_text()) for p in runs]
assert all(len(rows)==(3 if high else 7) for rows in results)
assert json.loads((runs[1]/'visit.json').read_text())['source_sha']=='37999f7ab31809bd8a9f3cabee3fa50c54f179d6'
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
except OSError:
    font = ImageFont.load_default(size=24)
report = dict(main_sha='0745bfe49b855ae48e0c1ef3305362b8e0c78a03',
              head_sha='37999f7ab31809bd8a9f3cabee3fa50c54f179d6',
              model='tencent/Hy-Nomas-Diffusers',pairs=[],visual_review='pending')
for index,(left,right) in enumerate(zip(*results)):
    assert left['workload_id']==right['workload_id'] and left['sha256']==right['sha256']
    width,height,fps=1280,704,24
    selected_frames=[0,frames//2,frames-1]
    paths=[root/Path(row['output']).relative_to('/task') for row in [left,right]]
    directory=out/left['workload_id'];directory.mkdir()
    commands=[];processes=[];decoded=[hashlib.sha256(),hashlib.sha256()]
    representatives={}
    for path in paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==left['sha256']
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
        video,=[s for s in meta['streams'] if s['codec_type']=='video']
        assert (video['width'],video['height'],int(video['nb_frames']))==(width,height,frames)
        assert video['avg_frame_rate']=='24/1'
        command=['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
        commands.append(command);processes.append(subprocess.Popen(command,stdout=subprocess.PIPE))
    try:
        for i in range(frames):
            raw=[p.stdout.read(width*height*3) for p in processes]
            assert all(len(x)==width*height*3 for x in raw) and raw[0]==raw[1]
            for h,b in zip(decoded,raw):h.update(b)
            if i in selected_frames:representatives[i]=[Image.frombytes('RGB',(width,height),b) for b in raw]
        assert all(p.stdout.read(1)==b'' for p in processes)
        assert all(p.wait(timeout=30)==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate();p.wait(timeout=10)
    canvas=Image.new('RGB',(2*width,3*(height+90)),'white');draw=ImageDraw.Draw(canvas)
    for row,(frame,images) in enumerate(representatives.items()):
        top=row*(height+90)
        for col,(label,im) in enumerate(zip(['main 0745bfe4','PR 37999f7a'],images)):
            x=col*width
            draw.text((x+12,top+8),f'Hy-Nomas | {left["prompt_id"]} prompt | {label}',font=font,fill='black')
            draw.text((x+12,top+38),f'BF16 {"high" if high else "lossless"} | 1280x704 | {frames}f/24fps | 8 steps | CFG1 | seed {left["seed"]}',font=font,fill='black')
            draw.text((x+12,top+65),f'frame {frame}, t={frame/fps:.3f}s',font=font,fill='black')
            canvas.paste(im,(x,top+90))
    canvas.save(directory/'comparison.png')
    entry=dict(workload_id=left['workload_id'],paths=[str(p.relative_to(root)) for p in paths],
               artifact_sha256=left['sha256'],artifact_exact=True,decoded_frames=frames,
               decoded_rgb_exact=True,decoded_rgb_sha256=[h.hexdigest() for h in decoded],
               all_frame_ssim=1.0,all_frame_psnr_db=None,psnr_note='positive infinity; byte-exact RGB',
               max_abs_error=0,representative_frames=selected_frames,timestamps=[i/fps for i in selected_frames],
               comparison=str((directory/'comparison.png').relative_to(root)),decode_commands=commands)
    (directory/'metrics.json').write_text(json.dumps(entry,indent=2)+'\n');report['pairs'].append(entry)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(pairs=len(report['pairs']),all_artifact_and_decoded_exact=True,path=str(out))))
