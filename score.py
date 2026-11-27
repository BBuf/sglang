"""Compute the SGLang image-consistency metrics for an actual main/head PNG pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sglang.multimodal_gen.test import test_utils as metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_png", type=Path)
    parser.add_argument("head_png", type=Path)
    parser.add_argument("--clip-model", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    metrics.CLIP_MODEL_NAME = str(args.clip_model.resolve(strict=True))
    a, b = [np.asarray(Image.open(path).convert("RGB")) for path in (args.main_png, args.head_png)]
    clip = metrics.compute_clip_similarity(metrics.compute_clip_embedding(a), metrics.compute_clip_embedding(b))
    print(json.dumps(dict(clip=float(clip), ssim=float(metrics.compute_ssim(b, a)), psnr_db=float(metrics.compute_psnr(b, a)), mean_abs_diff=float(metrics.compute_mean_abs_diff(b, a))), indent=2))


if __name__ == "__main__":
    main()
