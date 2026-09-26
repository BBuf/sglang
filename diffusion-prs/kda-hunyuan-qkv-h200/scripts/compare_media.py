#!/usr/bin/env python3
"""Compare every decoded output frame, retain hashes, and render authentic media."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--quality', choices=('lossless', 'high'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    is_video = args.before.suffix.lower() in ('.mp4', '.mov', '.webm')
    def frames(path):
        if is_video:
            return list(iio.imiter(path, plugin='FFMPEG'))
        return [np.array(Image.open(path).convert('RGB'))]
    baseline, candidate = frames(args.before), frames(args.after)
    assert len(baseline) == len(candidate) and len(baseline), 'Frame count mismatch'
    per_frame = []
    for i, (a, b) in enumerate(zip(baseline, candidate)):
        assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape)
        exact = bool(np.array_equal(a, b))
        psnr = math.inf if exact else float(peak_signal_noise_ratio(a, b, data_range=255))
        per_frame.append(dict(frame=i, pixel_exact=exact,
                              baseline_sha256=hashlib.sha256(a.tobytes()).hexdigest(), candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
                              ssim=1.0 if exact else float(structural_similarity(a, b, channel_axis=-1, data_range=255)),
                              psnr_db=psnr if math.isfinite(psnr) else 'inf',
                              max_abs_error=0 if exact else int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())))
    ssims = [row['ssim'] for row in per_frame]
    finite_psnr = [float(row['psnr_db']) for row in per_frame]
    before_hash, after_hash = file_hash(args.before), file_hash(args.after)
    floor_ssim, floor_psnr = (0.92, 24) if is_video else (0.95, 28)
    result = dict(quality=args.quality, media_type='video' if is_video else 'image', frames=len(per_frame),
                  byte_exact=before_hash == after_hash, pixel_exact=all(row['pixel_exact'] for row in per_frame),
                  before_sha256=before_hash, after_sha256=after_hash,
                  ssim_mean=float(np.mean(ssims)), ssim_min=min(ssims),
                  psnr_mean_db=float(np.mean(finite_psnr)) if all(math.isfinite(x) for x in finite_psnr) else 'inf',
                  psnr_min_db=min(finite_psnr) if math.isfinite(min(finite_psnr)) else 'inf',
                  required_ssim_min=floor_ssim, required_psnr_min_db=floor_psnr, per_frame=per_frame)
    result['qualified'] = result['byte_exact'] and result['pixel_exact'] if args.quality == 'lossless' else min(ssims) >= floor_ssim and min(finite_psnr) >= floor_psnr
    if is_video:
        streams = []
        for path in [args.before, args.after]:
            metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))
            streams.append(metadata['streams'])
        result['streams'] = streams
        video_streams = [next(stream for stream in arm if stream['codec_type'] == 'video') for arm in streams]
        assert all(video_streams[0].get(key) == video_streams[1].get(key) for key in ['width', 'height', 'r_frame_rate', 'nb_frames'])
        audio_streams = [[stream for stream in arm if stream['codec_type'] == 'audio'] for arm in streams]
        assert len(audio_streams[0]) == len(audio_streams[1]), 'Audio stream count mismatch'
        result['audio'] = []
        for index in range(len(audio_streams[0])):
            assert all(audio_streams[0][index].get(key) == audio_streams[1][index].get(key) for key in ['channels', 'sample_rate'])
            decoded = [subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-map', f'0:a:{index}', '-vn', '-acodec', 'pcm_f32le', '-f', 'f32le', '-']) for path in [args.before, args.after]]
            samples = [np.frombuffer(value, dtype='<f4') for value in decoded]
            assert samples[0].shape == samples[1].shape and all(np.isfinite(x).all() for x in samples)
            exact = bool(np.array_equal(*samples))
            result['audio'].append(dict(stream=index, decoded_exact=exact, baseline_sha256=hashlib.sha256(decoded[0]).hexdigest(), candidate_sha256=hashlib.sha256(decoded[1]).hexdigest(), samples=int(samples[0].size), max_abs_error=float(np.abs(samples[0]-samples[1]).max()) if samples[0].size else 0))
            result['qualified'] &= exact

    for label, path in [('baseline', args.before), ('candidate', args.after)]:
        shutil.copy2(path, args.output / (label + path.suffix))
    indices = sorted(set(np.linspace(0, len(baseline)-1, min(4, len(baseline)), dtype=int).tolist()))
    width = 512
    height = round(baseline[0].shape[0] * width / baseline[0].shape[1])
    canvas = Image.new('RGB', (width * len(indices), (height + 32) * 2), 'white')
    draw = ImageDraw.Draw(canvas)
    for r, (label, source) in enumerate([('Before', baseline), ('After', candidate)]):
        for c, index in enumerate(indices):
            image = Image.fromarray(source[index]).resize((width, height))
            canvas.paste(image, (width*c, (height+32)*r+32))
            draw.text((width*c+8, (height+32)*r+8), f'{label} | frame {index}', fill='black')
    canvas.save(args.output/'comparison.png')
    if is_video:
        previews=[]
        for index in np.linspace(0,len(baseline)-1,min(32,len(baseline)),dtype=int):
            pair=Image.new('RGB',(width*2,height+24),'white')
            d=ImageDraw.Draw(pair)
            for col,(label,source) in enumerate([('Before',baseline),('After',candidate)]):
                pair.paste(Image.fromarray(source[index]).resize((width,height)),(col*width,24))
                d.text((col*width+8,6),label,fill='black')
            previews.append(pair)
        previews[0].save(args.output/'comparison.gif',save_all=True,append_images=previews[1:],duration=160,loop=0)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({key:value for key,value in result.items() if key!='per_frame'},indent=2))
    return 0 if result['qualified'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
