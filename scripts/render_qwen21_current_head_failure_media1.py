"""Render primary RGBA evidence, explicitly retaining failed 1024 cleanup."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
HEAD = '188169fa0db0bb2d524d56a97184590f948cca07'
PLAN_HASHES = {512: 'fee7c53db5f9f63ac936d27283a827509411590158fd96b9797c61551a3975ad', 1024: 'feea9750bffab8c879912db5ee41c9467fa72389e187f9ca12014409fa7e905c'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


destination = EVIDENCE / 'media/qwen21-current-head-cleanup-failed-188169fa'
assert not destination.exists()
primary_receipts = {}
for size in (512, 1024):
    path = EVIDENCE / 'preflight' / f'qwen21-current-head-{size}-primary1-admission.json'
    admission = read(path)
    assert admission['source_sha'] == HEAD and admission['passed'] == (size == 512)
    assert admission['all_samples_retained'] and admission['source_and_ownership_audit_passed']
    assert all(case['primary_passed'] and case['cleanup_passed'] == (size == 512)
               for case in admission['cases'].values())
    primary_receipts[size] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), passed=admission['passed'])

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path, 23)
small = ImageFont.truetype(font_path, 19)
records = []
for size in (512, 1024):
    run = f'qwen21-current-head-{size}-primary1'
    plan_path = EVIDENCE / 'plans' / (run + '.json')
    assert digest(plan_path) == PLAN_HASHES[size]
    plan = read(plan_path)
    archive = read(EVIDENCE / 'checkpoints' / (run + '-local-verification.json'))
    assert archive['all_member_hashes_verified'] and archive['all_members_extracted']
    for task in ('t2i', 'edit'):
        contract_path = ROOT / 'inputs' / plan['contracts'][task]
        assert digest(contract_path) == plan['contract_sha256'][task]
        contract = read(contract_path)
        for index in range(3):
            pair = []
            for arm_index, source in [(0, 'main'), (2, 'current')]:
                name = f'{run}-{arm_index:02d}-{source}'
                metadata_path = EVIDENCE / 'block-runs' / name / 'metadata.json'
                metadata = read(metadata_path)
                assert metadata['status'] == 'complete' and metadata['plan'] == plan
                assert metadata['source_sha'] == plan['configurations'][source]['head']
                assert metadata['source_diff_sha256'] == hashlib.sha256(b'').hexdigest()
                folder = EVIDENCE / 'qualification' / f'sgl-qwen21-base-{task}-{name}-high'
                rows = read(folder / 'results.json')
                assert len(rows) == 4 and rows[0]['sha256'] == rows[3]['sha256']
                row = rows[index]
                request_name = Path(row['request']).name
                request_path = folder / request_name / 'request.json'
                request = read(request_path)
                payload = request['payload']
                assert payload['prompt'] == contract['prompts'][index]['prompt']
                assert payload['size'] == f'{size}x{size}' and payload['quality'] == 'high'
                for key in ('seed', 'num_inference_steps', 'true_cfg_scale', 'generator_device'):
                    assert str(payload[key]) == str(contract[key]), key
                path = folder / request_name / 'output.png'
                assert digest(path) == row['sha256']
                with Image.open(path) as opened:
                    assert opened.mode == 'RGBA' and opened.size == (size, size)
                    image = opened.copy()
                pair.append(dict(source=source, head=metadata['source_sha'], path=path,
                    sha256=row['sha256'], image=image, request=request_name,
                    metadata=str(metadata_path.relative_to(ROOT)), metadata_sha256=digest(metadata_path),
                    request_sha256=digest(request_path), payload=payload,
                    reference_image=request['reference_image']))
            assert pair[0]['request'] == pair[1]['request']
            assert pair[0]['sha256'] == pair[1]['sha256']
            assert pair[0]['image'].tobytes() == pair[1]['image'].tobytes()
            normalized = [{k: v for k, v in item['payload'].items() if k != 'perf_dump_path'} for item in pair]
            assert normalized[0] == normalized[1]
            assert pair[0]['reference_image'] == pair[1]['reference_image']
            canvas = Image.new('RGBA', (2 * size + 80, size + 246), '#f8fafc')
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 14), f'Qwen-Image 2.1 | {task.upper()} | {size}x{size} | BF16 high', font=font, fill='#111827')
            draw.text((24, 50), '50 steps | seed 1101 | CFG 1 | RTX PRO 6000 Blackwell', font=small, fill='#334155')
            draw.text((24, 78), 'Cleanup: ' + ('512 PASSED; 1024 FAILED separately' if size == 512 else '1024 FAILED; all primary samples retained'), font=small, fill='#9f1239')
            for x, item in zip((24, size + 56), pair):
                label = 'Main' if item['source'] == 'main' else 'PR current'
                draw.text((x, 126), label + ': ' + item['head'][:12], font=font, fill='#111827')
                canvas.paste(item['image'], (x, 170))
                assert canvas.crop((x, 170, x + size, 170 + size)).tobytes() == item['image'].tobytes()
            draw.text((24, size + 204), pair[0]['request'] + ' | Exact RGB and alpha; identical PNG bytes', font=small, fill='#334155')
            destination.mkdir(exist_ok=True)
            target = destination / f'{task}-{size}-{pair[0]["request"]}-main-vs-current.png'
            assert not target.exists()
            canvas.save(target)
            records.append(dict(size=size, task=task, quality='high', request=pair[0]['request'],
                prompt=contract['prompts'][index]['prompt'], panel=str(target.relative_to(ROOT)),
                panel_sha256=digest(target), contract_sha256=digest(contract_path),
                pixel_exact_RGBA=True, PNG_byte_exact=True,
                source_images=[{k: str(v.relative_to(ROOT)) if k == 'path' else v
                                for k, v in item.items() if k not in ('image', 'payload')}
                               for item in pair]))
assert len(records) == 12
result = dict(source_sha=HEAD, primary_admissions=primary_receipts, cleanup_passed=False,
    script_sha256=digest(Path(__file__)), rows=records, visual_review_pending=True,
    composition='Complete original RGBA images pasted at 1:1 without a mask, conversion or resampling; labels outside image bounds.',
    scope='Actual primary qualification images: short, long and Chinese prompts for T2I/edit at512/1024. 512 gates passed; 1024 cleanup failed. Current-head full quality transitions and BCG diagnostics have not run. These images do not qualify the current head or establish BCG performance.')
(destination / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(panels=len(records), output=str(destination), visual_review_pending=True)))
