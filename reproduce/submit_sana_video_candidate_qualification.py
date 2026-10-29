"""Save native HTTP videos and decode every frame for qualification, never timing admission."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import requests
from PIL import Image


def decode_video(path, contract):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    assert tuple(metadata["size"]) == (contract["width"], contract["height"]), metadata
    assert abs(metadata["fps"] - contract["fps"]) < 1e-6
    hashes = []
    digest = hashlib.sha256()
    for index, frame in enumerate(reader):
        assert len(frame) == contract["width"] * contract["height"] * 3
        hashes.append(hashlib.sha256(frame).hexdigest())
        digest.update(frame)
        if index in [0, 40, 80]:
            Image.frombytes("RGB", metadata["size"], frame).save(path.parent / f"frame-{index:03d}.png")
    assert len(hashes) == contract["num_frames"], len(hashes)
    return dict(metadata=metadata, decoded_rgb_sha256=digest.hexdigest(), frame_sha256=hashes, frames=len(hashes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["sgl", "omni"])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expected-diff-sha256", required=True)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    assert contract["batch"] == contract["concurrency"] == 1
    assert contract["physical_gpus"] == [0] and contract["task"] == "t2v"
    assert not args.profile or args.arm == "sgl"
    assert args.arm == "sgl"
    source = args.repo
    assert subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() == contract["pins"][args.arm]
    patch = subprocess.check_output(["git", "-C", source, "diff", "HEAD", "--binary", "--full-index"])
    assert hashlib.sha256(patch).hexdigest() == args.expected_diff_sha256
    folder = Path("/task/outputs/qualification") / f"{args.arm}-sana-video-{args.label}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "contract.json").write_bytes(contract_bytes)
    (folder / "metadata.json").write_text(json.dumps(dict(arm=args.arm, benchmark=False, profiled=args.profile, source_diff_sha256=args.expected_diff_sha256, source_sha=contract["pins"][args.arm], input_sha256=hashlib.sha256(contract_bytes).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")
    prompts = {row["id"]: row for row in contract["prompts"]}
    sequence = contract["sequence"][:1] if args.profile else contract["sequence"]
    base = "http://127.0.0.1:" + ("31050" if args.arm == "sgl" else "31051")
    rows = []
    with requests.Session() as session:
        for index, prompt_id in enumerate(sequence):
            target = folder / f"{index:02d}-{prompt_id}"
            target.mkdir()
            payload = {key: contract[key] for key in ["width", "height", "num_frames", "fps", "num_inference_steps", "guidance_scale", "seed", "negative_prompt"]}
            payload.update(prompt=prompts[prompt_id]["prompt"], model=contract["model_path"])
            if args.arm == "sgl":
                payload["prompt"] += " motion score: 30."
                payload.update(generator_device="cpu", max_sequence_length=300, quality="lossless", output_compression=100, x264_preset="ultrafast", perf_dump_path=str(target / "perf.json"), output_path=str(target / "native-output"), enhance_prompt=False)
                if args.profile:
                    payload.update(profile=True, num_profiled_timesteps=3)
                endpoint = "/v1/videos"
            else:
                payload["extra_params"] = json.dumps(contract["encoding"]["omni_extra_params"])
                endpoint = "/v1/videos/sync"
            (target / "request.json").write_text(json.dumps(dict(endpoint=endpoint, payload=payload), indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(dict(status="submitting", target=str(target))), flush=True)
            started = time.monotonic()
            if args.arm == "sgl":
                response = session.post(base + endpoint, json=payload, timeout=7200)
                (target / "create-response.json").write_text(response.text)
                response.raise_for_status()
                job = response.json()
                video_id = job["id"]
                polls = 0
                while job["status"] not in ["completed", "failed", "cancelled"]:
                    assert time.monotonic() - started < 7200
                    time.sleep(0.01)
                    response = session.get(base + "/v1/videos/" + video_id, timeout=30)
                    response.raise_for_status()
                    job = response.json()
                    polls += 1
                (target / "terminal-response.json").write_text(json.dumps(job, indent=2) + "\n")
                assert job["status"] == "completed", job
                response = session.get(base + "/v1/videos/" + video_id + "/content", timeout=120)
            else:
                response = session.post(base + endpoint, data=payload, timeout=7200)
                polls = 0
            elapsed = time.monotonic() - started
            (target / "response-headers.json").write_text(json.dumps(dict(response.headers), indent=2) + "\n")
            if not response.ok:
                (target / "error-response.txt").write_text(response.text)
            response.raise_for_status()
            assert "video/mp4" in response.headers.get("Content-Type", ""), response.headers
            path = target / "output.mp4"
            path.write_bytes(response.content)
            decoded = decode_video(path, contract)
            row = dict(request=str(target), sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content), http_seconds=elapsed, poll_count=polls, benchmark=False, profiled=args.profile, **decoded)
            rows.append(row)
            (folder / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ["request", "sha256", "decoded_rgb_sha256", "http_seconds", "frames"]}), flush=True)
    if not args.profile:
        assert rows[0]["sha256"] == rows[-1]["sha256"], "Native repeated short MP4 changed"
        assert rows[0]["decoded_rgb_sha256"] == rows[-1]["decoded_rgb_sha256"]


if __name__ == "__main__":
    main()
