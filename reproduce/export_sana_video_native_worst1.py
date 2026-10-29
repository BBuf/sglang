"""Extract the known failed native frames with the original common CPU decoder."""
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

root = Path('/task')
assert json.loads((root/'evidence/preflight/sana-video-depthwise-v1-paired1-admission.json').read_text())['passed']
assert not (root/'sana-video-depthwise-final-primary-cleanup1-status.json').exists()
decoder = Path(imageio_ffmpeg.get_ffmpeg_exe())
decoder_sha = hashlib.sha256(decoder.read_bytes()).hexdigest()
assert decoder_sha == '700073daef5c23bbcb18c2eae60553a454a5221ec19b4a88c8c367a664671a7c'
out = root/'outputs/diagnostics/sana-video-depthwise-cross-worst1'
out.mkdir(exist_ok=False)
sources = []
for arm, head in [('sgl','af21e2b989e482cfdce39689126562f2209d53ac'),
                  ('omni','817d4c17feb702289a2d2b2cbacfd594c262b7a7')]:
    folder = root/'outputs/qualification'/(arm+'-sana-video-t2v-native1')
    assert json.loads((folder/'metadata.json').read_text())['source_sha']==head
    result = next(x for x in json.loads((folder/'results.json').read_text())
                  if Path(x['request']).name=='02-chinese')
    video = folder/'02-chinese/output.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest()==result['sha256']
    reader = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    meta = next(reader)
    assert tuple(meta['size'])==(832,480) and meta['fps']==16
    hashes, images = [], []
    for index, frame in enumerate(reader):
        digest = hashlib.sha256(frame).hexdigest()
        assert digest==result['frame_sha256'][index]
        hashes.append(digest)
        if index in [26,27,28]:
            path = out/f'{arm}-frame-{index:03d}.png'
            Image.frombytes('RGB',(832,480),frame).save(path)
            images.append(dict(frame=index,path=path.name,RGB_sha256=digest,
                               png_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(hashes)==81 and len(images)==3
    sources.append(dict(arm=arm,source_sha=head,video=str(video),video_sha256=result['sha256'],
                        all81_original_frame_hashes_verified=True,images=images))
manifest = dict(decoder=str(decoder),decoder_sha256=decoder_sha,sources=sources,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                no_new_generation=True,CPU_decode_only=True,
                note='Original archived MP4s; all81 frames match the original common decoder hashes. No model or GPU request.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(out),sources=len(sources),images=sum(len(x['images']) for x in sources))))
