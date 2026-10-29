"""Arrange actual final/main decoded frames; never synthesize model output."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
head = 'e0d694593a65cb5b463c4011b84ee50abefa5e43'
base = 'af21e2b989e482cfdce39689126562f2209d53ac'
audit = json.loads((evidence/'preflight/sana-video-ln-final-compatibility1-raw-audit.json').read_text())
assert audit['passed']
plan_path = evidence/'plans/sana-video-ln-final-compatibility1-plan-v2.json'
plan = json.loads(plan_path.read_text())
assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==audit['plan_sha256']
assert plan['final_head']==head
out = evidence/'media/sana-video-ln-final-e0d69459'
out.mkdir(exist_ok=False)
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path,27)
small = ImageFont.truetype(font_path,21)
gap, width, height, header, row_label = 24,832,480,140,37
manifest = []
for prompt, index in [('short',0),('long',1),('chinese',2)]:
    sources = []
    for arm, ordinal, expected in [('main_eager',0,base),('final_eager',1,head)]:
        label = f'sgl-sana-video-sana-video-ln-final-compatibility1-{ordinal:02d}-{arm}'
        folder = evidence/'qualification'/label
        meta = json.loads((folder/'metadata.json').read_text())
        assert meta['source_sha']==expected and meta['source_diff_sha256']==hashlib.sha256(b'').hexdigest()
        result = json.loads((folder/'results.json').read_text())[index]
        assert result['prompt_id']==prompt and result['quality']=='lossless'
        request = folder/Path(result['request']).name
        video = request/'output.mp4'
        assert hashlib.sha256(video.read_bytes()).hexdigest()==result['sha256']
        sources.append((request,result,expected))
    assert sources[0][1]['sha256']==sources[1][1]['sha256']
    assert sources[0][1]['frame_sha256']==sources[1][1]['frame_sha256']
    canvas = Image.new('RGB',(2*width+3*gap,header+3*(height+row_label)+gap),(248,249,251))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap,12),'SANA-Video_2B_480p | Native BF16 DiT/encoder, FP32 VAE',font=font,fill='black')
    draw.text((gap,49),'832 x 480 | 81 frames, 16 fps | 50 steps | CFG 6 | CPU seed 1101 | '+prompt+' prompt',font=small,fill='black')
    draw.text((gap,85),'main '+base[:12],font=font,fill='black')
    draw.text((2*gap+width,85),'PR '+head[:12],font=font,fill='black')
    for row_index, frame in enumerate([0,40,80]):
        y = header+row_index*(height+row_label)
        for column,(folder,result,_) in enumerate(sources):
            path = folder/f'frame-{frame:03d}.png'
            image = Image.open(path).convert('RGB')
            assert image.size==(width,height)
            assert hashlib.sha256(image.tobytes()).hexdigest()==result['frame_sha256'][frame]
            x = gap+column*(width+gap)
            draw.text((x,y),f'Frame {frame:03d} | {frame/16:.3f} s | exact RGB',font=small,fill='black')
            canvas.paste(image,(x,y+row_label))
    panel = out/(prompt+'-main-vs-pr.png')
    canvas.save(panel)
    manifest.append(dict(prompt=prompt,panel=str(panel.relative_to(root)),panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                         frames=[0,40,80],times_seconds=[0,2.5,5],MP4_exact=True,all81frames_exact=True,
                         inputs=[dict(arm=arm,source_sha=sha,video=str((folder/'output.mp4').relative_to(root)),sha256=result['sha256'])
                                 for arm,(folder,result,sha) in zip(['main','PR'],sources)]))
(out/'manifest.json').write_text(json.dumps(dict(rows=manifest,visually_inspected=False,scope='Real main/PR lossless eager requests; representative timestamps only. Whole-video motion inspection is not implied. Chinese crossOmni quality failure retained.'),indent=2)+'\n')
print(out)
