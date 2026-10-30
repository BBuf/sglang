"""Compose isolated Joy repair media only from admitted, archived native requests."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
run = 'joy-input-lifetime-repair1-diagnostics1'
destination = evidence/'media/joy-input-lifetime-repair1-03368cf1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan_path = evidence/'plans'/(run+'.json')
plan = read(plan_path)
assert digest(plan_path) == '30dc276d3673fb79f4a0311d3717fa1d40c44d292ccef137d40bb44da2938a29'
admission_path = evidence/'preflight'/(run+'-admission.json')
admission = read(admission_path)
assert admission['passed'] and admission['all_raw_outputs_rehashed'] == 76
assert admission['plan_sha256'] == digest(plan_path)
assert admission['source_sha'] == '03368cf1c325e526184b8fb223443e1ae931b13c'
for shape in ('square', 'wide'):
    assert read(evidence/'preflight'/f'joy-input-lifetime-repair1-{shape}-primary1-admission.json')['passed']
verified = read(evidence/'checkpoints'/(run+'-local-verification.json'))
assert verified['all_member_hashes_verified'] and verified['all_members_extracted']
destination.mkdir(exist_ok=False)
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path, 24)
small = ImageFont.truetype(font_path, 21)
records = []
for shape in plan['shapes']:
    contract_path = root/Path(plan['contracts'][shape]).relative_to('/task')
    contract = read(contract_path)
    width, height = contract['width'], contract['height']
    # High is the measured optimization lane. One lossless example per shape
    # complements the full two-phase restoration audit without hiding repeats.
    selections = [('1', 'high', i) for i in range(3)]+[('2', 'lossless', 0)]
    for phase, quality, index in selections:
        pair = []
        for source in ('main', 'current'):
            name = run+'-'+shape+'-'+source
            block = evidence/'block-runs'/name
            metadata_path = block/'metadata.json'
            metadata = read(metadata_path)
            chain = next(x for x in admission['source_chains'] if x['shape'] == shape and x['source'] == source)
            assert digest(metadata_path) == chain['metadata_sha256']
            assert metadata['source_sha'] == plan['sources'][source]['head'] == chain['source_sha']
            assert metadata['source_diff_sha256'] == hashlib.sha256(b'').hexdigest()
            folder = evidence/'qualification'/('sgl-joy-'+name+'-'+phase+'-'+quality)
            assert digest(folder/'contract.json') == digest(contract_path)
            rows = read(folder/'results.json')
            request_name = Path(rows[index]['request']).name
            request = read(folder/request_name/'request.json')
            assert request['form']['prompt'] == contract['prompts'][index]['prompt']
            assert request['form']['quality'] == quality
            path = folder/request_name/'output.png'
            assert digest(path) == rows[index]['sha256']
            pair.append(dict(source=source, head=metadata['source_sha'], path=path,
                             sha256=rows[index]['sha256'], request=request_name,
                             metadata=str(metadata_path.relative_to(root))))
        assert pair[0]['request'] == pair[1]['request']
        metrics = next(x for x in admission['quality_rows'] if x['shape'] == shape and x['phase'] == phase
                       and x['quality'] == quality and x['request'] == pair[0]['request'])
        assert metrics['passed'] and metrics['main_sha256'] == pair[0]['sha256']
        assert metrics['current_sha256'] == pair[1]['sha256']
        images = [Image.open(x['path']).convert('RGB') for x in pair]
        assert all(im.size == (width, height) for im in images)
        a, b = map(np.asarray, images)
        exact = bool(np.array_equal(a, b))
        ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255))
        psnr = None if exact else float(peak_signal_noise_ratio(a, b, data_range=255))
        assert (exact, ssim, psnr) == (metrics['pixel_identical'], metrics['ssim'], metrics['psnr_db'])
        canvas = Image.new('RGB', (2*width+80, height+212), '#f8fafc')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), f'JoyAI Image Edit | {width}x{height} | 30 steps | seed 1101', font=font, fill='#111827')
        draw.text((24, 50), f'Native BF16 DiT / encoder / VAE | CFG 4 | {quality} | {shape}', font=small, fill='#334155')
        for x, item, im in zip((24, width+56), pair, images):
            label = 'Main' if item['source'] == 'main' else 'Lifetime repair'
            draw.text((x, 96), label+': '+item['head'][:12], font=font, fill='#111827')
            canvas.paste(im, (x, 140))
            assert canvas.crop((x, 140, x+width, 140+height)).tobytes() == im.tobytes()
        psnr_text = 'infinite' if psnr is None else f'{psnr:.4f} dB'
        draw.text((24, height+170), f'{pair[0]["request"]} | SSIM {ssim:.6f} | PSNR {psnr_text}', font=small, fill='#334155')
        target = destination/(shape+'-'+quality+'-'+pair[0]['request']+'-main-vs-current.png')
        canvas.save(target)
        records.append(dict(shape=shape, phase=phase, quality=quality, request=pair[0]['request'],
            prompt=contract['prompts'][index]['prompt'], panel=str(target.relative_to(root)), panel_sha256=digest(target),
            source_images=[{**x, 'path':str(x['path'].relative_to(root))} for x in pair],
            contract_sha256=digest(contract_path), pixel_identical=exact, ssim=ssim, psnr_db=psnr,
            composition='Complete source RGB images pasted at 1:1 without resampling; labels outside image bounds.'))
assert len(records) == 12
result = dict(source_sha=admission['source_sha'], official_PR_head='b08c24e9630c63790c9f86b48d6c7fc10a0c4301',
              repair_is_official_PR_head=False, actual_diagnostics_admission_sha256=digest(admission_path),
              plan_sha256=digest(plan_path), rows=records, visual_review_pending=True,
              script_sha256=digest(Path(__file__)),
              scope='Three high prompts and one returned-lossless example per shape; all36quality comparisons remain in the diagnostic admission. Numerical admission does not replace visual inspection.')
(destination/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(panels=len(records), output=str(destination), visual_review_pending=True)))
