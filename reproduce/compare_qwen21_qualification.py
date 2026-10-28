"""Audit original Qwen PNGs and compare RGB and native alpha independently."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, candidate, channel_axis):
    exact = bool(np.array_equal(reference, candidate))
    return dict(pixel_exact=exact,
                max_absolute_error=int(np.max(np.abs(reference.astype(np.int16)-candidate.astype(np.int16)))),
                ssim=float(structural_similarity(reference, candidate, channel_axis=channel_axis, data_range=255)),
                psnr_db=None if exact else float(peak_signal_noise_ratio(reference, candidate, data_range=255)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate')
    parser.add_argument('--reference', action='append', required=True)
    parser.add_argument('--root', type=Path, default=Path('/task/outputs/qualification'))
    parser.add_argument('--require-pixel-identical', action='store_true')
    args = parser.parse_args()
    folder = args.root/args.candidate
    contract_bytes = (folder/'contract.json').read_bytes()
    contract = json.loads(contract_bytes)
    rows = json.loads((folder/'results.json').read_text())
    assert len(rows) == 4 and all(not row['profiled'] for row in rows)
    expected_names = [f'{i:02d}-{prompt["id"]}' for i,prompt in
                      enumerate(contract['prompts']+[contract['prompts'][0]])]
    results = []
    for ref in args.reference:
        reference_folder = args.root/ref
        assert (reference_folder/'contract.json').read_bytes() == contract_bytes
        reference_rows = json.loads((reference_folder/'results.json').read_text())
        assert len(reference_rows) == len(rows)
        for name, row, reference_row in zip(expected_names, rows, reference_rows):
            source = folder/name/'output.png'
            reference = reference_folder/name/'output.png'
            assert Path(row['request']).name == Path(reference_row['request']).name == name
            assert digest(source) == row['sha256'] and digest(reference) == reference_row['sha256']
            with Image.open(source) as candidate_image, Image.open(reference) as reference_image:
                modes = [reference_image.mode, candidate_image.mode]
                assert reference_image.size == candidate_image.size == (contract['width'], contract['height'])
                # Conversion only normalizes the comparison view. Originals are
                # untouched, and missing native alpha cannot pass this gate.
                native_alpha = all('A' in image.getbands() for image in (reference_image,candidate_image))
                candidate = np.asarray(candidate_image.convert('RGBA'))
                baseline = np.asarray(reference_image.convert('RGBA'))
            parts = dict(rgb=metrics(baseline[:,:,:3], candidate[:,:,:3], 2),
                         alpha=metrics(baseline[:,:,3], candidate[:,:,3], None))
            gate = contract['quality_gate']['cross_framework']
            passed = native_alpha and all(value['pixel_exact'] or
                (value['ssim'] >= gate[f'{key}_ssim_min'] and value['psnr_db'] >= gate[f'{key}_psnr_db_min'])
                for key,value in parts.items())
            pixel_exact = all(value['pixel_exact'] for value in parts.values())
            results.append(dict(prompt=name, reference=ref, modes=modes, native_alpha_preserved=native_alpha,
                candidate_sha256=row['sha256'], reference_sha256=reference_row['sha256'],
                pixel_exact=pixel_exact, **parts, pass_gate=passed and (pixel_exact or not args.require_pixel_identical)))
    result = dict(candidate=args.candidate, contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
                  thresholds=contract['quality_gate'], results=results, all_pass=all(row['pass_gate'] for row in results),
                  visual_review_pending=True)
    (args.root/(args.candidate+'-quality.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['all_pass']:
        raise SystemExit('Quality gate failed; preserve originals and do not benchmark this configuration.')


if __name__ == '__main__':
    main()
