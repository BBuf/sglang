"""Display the known native cross-framework failure from original MP4s."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
quality_path = evidence/'preflight/sana-video-native-cross-quality1.json'
quality = json.loads(quality_path.read_text())
row = next(x for x in quality['rows'] if x['request']=='02-chinese')
failed = [x for x in row['frames'] if not x['passed']]
assert [x['index'] for x in failed] == [26, 27, 28]
width, height, count = 832, 480, 81
ffmpeg = shutil.which('ffmpeg')
assert ffmpeg
sources = []
for arm, source_sha in [('sgl','af21e2b989e482cfdce39689126562f2209d53ac'),
                        ('omni','817d4c17feb702289a2d2b2cbacfd594c262b7a7')]:
    folder = evidence/'qualification'/(arm+'-sana-video-t2v-native1')
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['source_sha']==source_sha
    result = next(x for x in json.loads((folder/'results.json').read_text())
                  if Path(x['request']).name=='02-chinese')
    video = folder/'02-chinese/output.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest()==result['sha256']
    command = [ffmpeg,'-nostdin','-v','error','-threads','1','-i',str(video),
               '-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    decoded = subprocess.run(command,check=True,capture_output=True).stdout
    size = width*height*3
    assert len(decoded)==count*size
    hashes = [hashlib.sha256(decoded[i*size:(i+1)*size]).hexdigest() for i in range(count)]
    assert hashes==result['frame_sha256'], 'Local decoder differs from the original frame bytes; do not publish'
    sources.append(dict(arm=arm,source_sha=source_sha,video=str(video.relative_to(root)),
                        video_sha256=result['sha256'],all81_original_frame_hashes_verified=True,
                        frames={i:decoded[i*size:(i+1)*size] for i in [26,27,28]}))

out = evidence/'media/sana-video-native-cross-worst1'
out.mkdir(exist_ok=False)
gap, header, caption = 24, 135, 42
canvas = Image.new('RGB',(2*width+3*gap,header+3*(height+caption)+gap),(248,249,251))
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25)
small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
draw.text((gap,10),'SANA-Video native Chinese prompt | Known cross-framework quality failure',font=font,fill='black')
draw.text((gap,48),'832 x 480 | 81 frames, 16 fps | 50 steps | CFG 6 | seed 1101 | BF16 DiT/TE, FP32 VAE',font=small,fill='black')
draw.text((gap,88),'SGLang main af21e2b989e4',font=font,fill='black')
draw.text((2*gap+width,88),'vLLM-Omni 817d4c17feb7',font=font,fill='black')
for index, metric in enumerate(failed):
    frame = metric['index']
    y = header+index*(height+caption)
    for column, source in enumerate(sources):
        image = Image.frombytes('RGB',(width,height),source['frames'][frame])
        path = out/f'{source["arm"]}-frame-{frame:03d}.png'
        image.save(path)
        x = gap+column*(width+gap)
        draw.text((x,y),f'Frame {frame} | {frame/16:.4f} s | PSNR {metric["psnr_db"]:.4f} dB < 24',font=small,fill='black')
        canvas.paste(image,(x,y+caption))
panel = out/'chinese-worst-frames.png'
canvas.save(panel)
manifest = dict(sources=[{k:v for k,v in x.items() if k!='frames'} for x in sources],
                failed_frames=failed,quality_report_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                local_decoder=ffmpeg,local_decoder_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
                local_decoder_sha256=hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest(),
                original_decoder=quality['decoder'],original_decoder_sha256=quality['decoder_sha256'],
                decoder_note='Different local executable; every decoded frame was required to match the original common-decoder hashes before rendering.',
                panel=str(panel.relative_to(root)),panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                visual_review_pending=True,quality_failure_retained=True,
                scope='Original frozen native videos, no new generation or performance run. This panel is SGLang main versus Omni, not the required separate main/PR panels.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['panel','panel_sha256','local_decoder_version','quality_failure_retained']}))
