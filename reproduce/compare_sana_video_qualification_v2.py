"""Compare all native encoded frames through one decoder, retaining every failure."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def frames(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    return metadata, (np.frombuffer(value, dtype=np.uint8).reshape(height, width, 3) for value in reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("--root", type=Path, default=Path("/task/outputs/qualification"))
    args = parser.parse_args()
    for name in [args.candidate, args.reference]:
        assert Path(name).name == name
    candidate, reference = [args.root / name for name in [args.candidate, args.reference]]
    assert (candidate / "contract.json").read_bytes() == (reference / "contract.json").read_bytes()
    contract = json.loads((candidate / "contract.json").read_text())
    gates = contract["quality_gates"]["cross_framework"]
    crows, rrows = [json.loads((directory / "results.json").read_text()) for directory in [candidate, reference]]
    assert len(crows) == len(rrows) == 4
    name = args.candidate + "-vs-" + args.reference + "-v2"
    output = args.root / (name + "-quality.json")
    assert not output.exists()
    media = args.root / (name + "-decoded-contact-frames")
    media.mkdir(exist_ok=False)
    rows = []
    for cr, rr in zip(crows, rrows):
        request = Path(cr["request"]).name
        assert request == Path(rr["request"]).name
        cp, rp = [directory / request / "output.mp4" for directory in [candidate, reference]]
        cm, ci = frames(cp)
        rm, ri = frames(rp)
        assert cm["size"] == rm["size"] == (contract["width"], contract["height"])
        assert cm["fps"] == rm["fps"] == contract["fps"]
        frame_rows = []
        for index, (c, r) in enumerate(zip(ci, ri, strict=True)):
            exact = np.array_equal(c, r)
            mse = np.square(c.astype(np.float64) - r).mean()
            psnr = None if mse == 0 else float(10 * np.log10(255**2 / mse))
            ssim = float(structural_similarity(c, r, channel_axis=2, data_range=255))
            frame_rows.append(dict(index=index, seconds=index / contract["fps"], pixel_exact=exact, ssim=ssim, psnr_db=psnr, psnr_infinite=bool(mse == 0), passed=ssim >= gates["per_frame_ssim_min"] and (psnr is None or psnr >= gates["per_frame_psnr_db_min"])))
            if index in [0, 40, 80]:
                for arm, value in [("candidate", c), ("reference", r)]:
                    Image.fromarray(value).save(media / f"{request}-{arm}-{index:03d}.png")
        assert len(frame_rows) == 81
        finite_psnr = [row["psnr_db"] for row in frame_rows if row["psnr_db"] is not None]
        row = dict(request=request, encoded_bytes_equal=cp.read_bytes() == rp.read_bytes(), all_frames_pixel_exact=all(frame["pixel_exact"] for frame in frame_rows), all_frames_pass=all(frame["passed"] for frame in frame_rows), mean_ssim=float(np.mean([frame["ssim"] for frame in frame_rows])), min_ssim=min(frame["ssim"] for frame in frame_rows), mean_finite_psnr_db=float(np.mean(finite_psnr)) if finite_psnr else None, min_psnr_db=min(finite_psnr) if finite_psnr else None, infinite_psnr_frames=81 - len(finite_psnr), frames=frame_rows)
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "frames"}), flush=True)
    result = dict(candidate=args.candidate, reference=args.reference, contract_sha256=hashlib.sha256((candidate / "contract.json").read_bytes()).hexdigest(), decoder=imageio_ffmpeg.get_ffmpeg_exe(), decoder_sha256=hashlib.sha256(Path(imageio_ffmpeg.get_ffmpeg_exe()).read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), all_numeric_pass=all(row["all_frames_pass"] for row in rows), visual_review_pending=True, all_samples_retained=True, rows=rows)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
