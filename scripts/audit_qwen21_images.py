"""Audit native RGBA artifacts using the pinned Omni accuracy metric functions."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

torch.set_num_threads(1)
root = Path('/task')
sgl_run, omni_run, audit_name = sys.argv[1:4]
labels = json.loads((root/sys.argv[4]).read_text()) if len(sys.argv) == 5 else None
names = labels["names"] if labels else ["sgl", "omni"]
assert all(Path(n).name == n for n in (sgl_run, omni_run, audit_name))
out = root/'evidence/quality'/audit_name
out.mkdir(parents=True, exist_ok=False)
(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
(out/'audit-command.json').write_text(json.dumps(sys.argv)+'\n')
contract = json.loads((root/'inputs/qwen21-native-workloads.json').read_text())
metric_path = root/'repos/vllm-omni/tests/e2e/accuracy/helpers.py'
metric_names = {'_pil_to_batched_tensor', 'compute_image_ssim_psnr'}
nodes = [x for x in ast.parse(metric_path.read_text()).body
         if isinstance(x, ast.FunctionDef) and x.name in metric_names]
assert {x.name for x in nodes} == metric_names
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(metric_path), 'exec'))
metric_receipt = dict(source=str(metric_path), source_sha256=hashlib.sha256(metric_path.read_bytes()).hexdigest(),
    selected_functions=sorted(metric_names), torch=torch.__version__, torchmetrics=torchmetrics.__version__,
    compare_mode='RGBA', data_range=1.0, implementation='Original native test functions AST-executed; CPU only')
(out/'metric-receipt.json').write_text(json.dumps(metric_receipt, indent=2)+'\n')
rows = {}
attention_labels = {}
quality_labels = {}
for arm, run in zip(names, (sgl_run, omni_run), strict=True):
    folder = root/'runs'/run
    status = json.loads((folder/'status.json').read_text())
    assert status['status'] == 'raw_complete' and status['gpu_cleanup'] == 'idle_verified'
    rows[arm] = json.loads((folder/'outputs/results.json').read_text())
    visit = json.loads((folder/'visit.json').read_text())
    quality_labels[arm] = visit.get('quality', 'native-default')
    attention_labels[arm] = 'DiT cuDNN' if 'transformer=torch_cudnn_sdpa' in visit['server_args'][status['arm']] else 'native attention'
    assert len(rows[arm]) == 6 and all(x['native_steps_observed'] == 50 for x in rows[arm])

cases = [(f'{arm}-own-{prompt}', arm, first, arm, repeated, True)
         for arm in rows for prompt, first, repeated in [('short', 0, 4), ('edit', 3, 5)]]
cases += [(f'cross-{rows[names[0]][i]["prompt_id"]}', names[0], i, names[1], i,
           bool(labels and labels.get('cross_exact_required'))) for i in range(4)]
results = []
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
for name, left_arm, left_index, right_arm, right_index, exact_required in cases:
    directory = out/name
    directory.mkdir()
    selected = [(left_arm, left_index), (right_arm, right_index)]
    paths = [Path(rows[a][i]['output']) for a, i in selected]
    assert all(p.is_relative_to(root/'runs') for p in paths)
    images = []
    hashes = []
    for p, (arm, index) in zip(paths, selected):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == rows[arm][index]['sha256']
        hashes.append(digest)
        image = Image.open(p)
        image.load()
        assert image.mode == 'RGBA' and image.size == (1024, 1024), (p, image.mode, image.size)
        images.append(image)
    arrays = [np.asarray(im) for im in images]
    exact = bool(np.array_equal(*arrays))
    error = arrays[0].astype(np.float64)-arrays[1].astype(np.float64)
    ssim, psnr = compute_image_ssim_psnr(prediction=images[0], reference=images[1], compare_mode='RGBA')
    assert math.isfinite(ssim) and not math.isnan(psnr)
    gate = contract['quality']['cross_framework']
    passed = hashes[0] == hashes[1] and exact if exact_required else ssim >= gate['ssim_min'] and psnr >= gate['psnr_min_db']
    alpha = error[..., 3]
    metrics = dict(name=name, prompt_id=rows[left_arm][left_index]['prompt_id'],
        source_paths=list(map(str, paths)), artifact_sha256=hashes, artifact_exact=hashes[0] == hashes[1],
        decoded_rgba_exact=exact, max_abs_error=int(np.abs(error).max()), mean_abs_error=float(np.abs(error).mean()),
        mse=float(np.mean(error**2)), psnr_db=None if math.isinf(psnr) else psnr, ssim=ssim,
        alpha_mae_normalized=float(np.abs(alpha).mean()/255), alpha_max_abs=int(np.abs(alpha).max()),
        alpha_changed_pixels=int(np.count_nonzero(alpha)), rgb_mae=float(np.abs(error[..., :3]).mean()),
        exact_required=exact_required, numeric_passed=passed,
        alignment='No alignment, rescaling, cropping, normalization or filtering; raw RGBA metric. Infinite PSNR for identical output is null.',
        image_rendering='Only the evidence display is composited over a fixed checkerboard; numeric comparison uses original RGBA.')
    (directory/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    width = height = 1024
    yy, xx = np.indices((height, width))
    gray = np.where(((xx//32)+(yy//32))%2, 224, 255).astype(np.uint8)
    background = Image.fromarray(np.repeat(gray[..., None], 3, axis=-1)).convert('RGBA')
    canvas = Image.new('RGB', (2*width, height+80), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (im, (arm, index)) in enumerate(zip(images, selected)):
        label = labels['labels'][arm] if labels else ('SGLang main 0745bfe eager '+attention_labels[arm]+' '+quality_labels[arm] if arm == 'sgl' else 'vLLM-Omni aa2968e native default')
        draw.text((column*width+12, 5), f'{label} | request{index} | {metrics["prompt_id"]}', fill='black', font=font)
        draw.text((column*width+12, 33), 'Qwen-Image2.1 | 1024x1024 | 50steps | CPUseed1101 | CFG1 | RGBA', fill='black', font=font)
        canvas.paste(Image.alpha_composite(background, im).convert('RGB'), (column*width, 80))
    canvas.save(directory/'comparison.png')
    results.append(metrics)
cross = [r for r in results if r['name'].startswith('cross-')]
report = dict(numeric_passed=all(r['numeric_passed'] for r in results), results=results,
    visual_review_pending=True, worst_cross_ssim=min(cross, key=lambda r: r['ssim'])['name'],
    scope=labels['scope'] if labels else 'Three T2I prompts and one single-reference edit, one CPU seed, native50steps. Own repeats check non-first short and edit requests. No candidate or performance claim.')
(out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'results'}))
