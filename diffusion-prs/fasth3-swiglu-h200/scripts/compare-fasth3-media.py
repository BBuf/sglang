"""Audit every formal FastH3 video/audio output and render a comparison figure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

root = Path('/campaign/artifacts/fasth3')
p = argparse.ArgumentParser()
p.add_argument('--glob', default='fp8-swiglu-r1-*')
p.add_argument('--output', default='swiglu-output-comparison')
args = p.parse_args()
target = root / args.output
target.mkdir(exist_ok=True)
order = {'a1': 0, 'b1': 1, 'b2': 2, 'a2': 3}
paths = sorted(root.glob(args.glob + '/result.json'),
               key=lambda p: (p.parent.name.rsplit('-', 1)[0], order[p.parent.name.rsplit('-', 1)[1]]))
assert len(paths) >= 4 and len(paths) % 4 == 0, paths
records = []
reference_frames = reference_audio = None
all_exact = True
for path in paths:
    result = json.loads(path.read_text())
    assert not result['error'], result
    video = Path(result['output_artifacts'][0])
    sha = hashlib.file_digest(video.open('rb'), 'sha256').hexdigest()
    assert sha == result['output_sha256'][0]
    metadata = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)]))
    streams = {s['codec_type']: s for s in metadata['streams']}
    v = streams['video']
    assert (v['width'], v['height'], int(v['nb_frames']), v['r_frame_rate']) == (1344,768,243,'24/1')
    frames_text = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:v:0',
        '-an', '-f', 'framehash', '-hash', 'sha256', '-'], text=True)
    (target / f'{result["label"]}.framehash').write_text(frames_text)
    frames = [line.split(',')[-1].strip() for line in frames_text.splitlines() if line and not line.startswith('#')]
    assert len(frames) == 243
    audio_bytes = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:a:0',
        '-vn', '-acodec', 'pcm_f32le', '-f', 'f32le', '-'])
    samples = np.frombuffer(audio_bytes, dtype='<f4')
    assert np.isfinite(samples).all()
    if reference_frames is None:
        reference_frames, reference_audio = frames, samples.copy()
    frame_exact = frames == reference_frames
    audio_exact = np.array_equal(samples, reference_audio)
    all_exact &= frame_exact and audio_exact
    record = dict(label=result['label'], source_commit=(path.parent/'source.txt').read_text().splitlines()[0],
                  artifact_sha256=sha, frame_count=len(frames), video_stream=v, audio_stream=streams['audio'],
                  decoded_frames_exact_to_first_baseline=frame_exact,
                  changed_frame_indices=[i for i,(a,b) in enumerate(zip(frames,reference_frames)) if a!=b],
                  decoded_audio_exact_to_first_baseline=audio_exact,
                  decoded_audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
                  decoded_audio_samples_per_channel=len(samples)//int(streams['audio']['channels']),
                  audio_peak=float(np.abs(samples).max()), audio_rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2))))
    if frame_exact:
        record.update(pixel_max_error=0, frame_ssim_mean=1.0, frame_ssim_min=1.0,
                      frame_psnr_mean='infinity', frame_psnr_min='infinity')
    if audio_exact:
        record.update(audio_max_error=0, audio_rmse=0)
    records.append(record)
    print(result['label'], 'video exact:',frame_exact,'audio exact:',audio_exact,flush=True)

first_a = next(p for p in paths if p.parent.name.endswith('-a1'))
first_b = next(p for p in paths if p.parent.name.endswith('-b1'))
strips = []
for label, result_path in [('Baseline', first_a), ('Candidate', first_b)]:
    video = json.loads(result_path.read_text())['output_artifacts'][0]
    strip = target / f'{label.lower()}-strip.png'
    subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',video,
                    '-vf',"select='eq(n,0)+eq(n,80)+eq(n,161)+eq(n,242)',scale=480:-1,tile=4x1",
                    '-frames:v','1',str(strip)],check=True)
    strips.append((label,Image.open(strip).convert('RGB')))
width, height = strips[0][1].size
figure = Image.new('RGB',(width,2*(height+36)), 'white')
draw = ImageDraw.Draw(figure)
for index,(label,strip) in enumerate(strips):
    y=index*(height+36)
    draw.text((12,y+10),f'{label} | frames 0, 80, 161, 242 | 10.125 s, 24 fps',fill='black')
    figure.paste(strip,(0,y+36))
figure.save(target/'comparison.jpg',quality=94)
(target/'comparison.json').write_text(json.dumps(dict(all_exact=bool(all_exact),runs=records),indent=2))
raise SystemExit(0 if all_exact else 1)
