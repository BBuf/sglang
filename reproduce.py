"""Prepare the pinned native Nomas weight view and reproduce the delivered camera videos."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import requests


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(recipe_dir, canonical, text, target):
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if any(
        target.is_relative_to(src) or src.is_relative_to(target)
        for src in (canonical, text)
    ):
        raise ValueError("Choose a separate output directory for the native view")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as folder:
        staged = Path(folder) / "view"
        materialize(recipe_dir, canonical, text, staged)
        staged.rename(target)


def materialize(recipe_dir, canonical, text, target):
    manifest = json.loads((recipe_dir / "weights.json").read_text())
    trees = {"Hy-Nomas-Diffusers": canonical, "Qwen2.5-VL-7B-Instruct": text}
    target.mkdir(parents=True, exist_ok=False)
    for spec in manifest["configs"]:
        path = target / spec["path"]
        raw = (recipe_dir / "config" / spec["path"]).read_bytes()
        if digest(raw) != spec["sha256"]:
            raise ValueError(spec["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for spec in manifest["links"]:
        src = trees[spec["tree"]] / spec["file"]
        size = src.stat().st_size
        if size != spec["bytes"]:
            raise ValueError(src)
        # Hugging Face's small Git blobs carry their size in the digest preamble.
        checksum = (
            hashlib.sha256()
            if spec["check_type"] == "sha256"
            else hashlib.sha1(f"blob {size}\0".encode())
        )
        with src.open("rb") as reader:
            for block in iter(lambda: reader.read(8 << 20), b""):
                checksum.update(block)
        if checksum.hexdigest() != spec["digest"]:
            raise ValueError(f"Pinned source differs: {src}")
        path = target / spec["target"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(src)
    for spec in manifest["shards"]:
        dest = target / spec["file"]
        checksum = hashlib.sha256()
        total = 0
        with (canonical / spec["file"]).open("rb") as reader, dest.open("xb") as writer:
            header = reader.read(struct.unpack("<Q", reader.read(8))[0])
            if digest(header) != spec["source_header"]:
                raise ValueError(spec["file"])
            converted = {
                manifest["rename"].get(k, k): v for k, v in json.loads(header).items()
            }
            encoded = json.dumps(
                converted, separators=(",", ":"), ensure_ascii=False
            ).encode()
            encoded += b" " * (-len(encoded) % 8)
            if digest(encoded) != spec["target_header"]:
                raise ValueError(f"Renamed header differs for {dest.name}")
            writer.write(struct.pack("<Q", len(encoded)))
            writer.write(encoded)
            for block in iter(lambda: reader.read(8 << 20), b""):
                checksum.update(block)
                writer.write(block)
                total += len(block)
        if total != spec["payload_bytes"] or checksum.hexdigest() != spec["payload"]:
            raise ValueError(f"Weight payload differs for {dest.name}")


def generate(model, spec, out, port):
    base = f"http://127.0.0.1:{port}/v1/videos"
    out.mkdir(parents=True, exist_ok=False)
    records = []
    with requests.Session() as client:
        client.trust_env = False
        for i, item in enumerate(spec["requests"]):
            dest = out / f"{i:02d}"
            dest.mkdir()
            body = dict(
                item["body"],
                model=str(model),
                perf_dump_path=str((dest / "perf.json").resolve()),
            )
            start = time.monotonic()
            resp = client.post(base, json=body, timeout=240)
            resp.raise_for_status()
            job = resp.json()
            url = base + "/" + job["id"]
            while job["status"] not in ("completed", "failed", "canceled", "cancelled"):
                if time.monotonic() - start > 1800:
                    raise TimeoutError(job["id"])
                time.sleep(0.1)
                resp = client.get(url, timeout=90)
                resp.raise_for_status()
                job = resp.json()
            if job["status"] != "completed":
                raise RuntimeError(job)
            resp = client.get(url + "/content", timeout=240)
            resp.raise_for_status()
            elapsed = time.monotonic() - start
            path = dest / "output.mp4"
            path.write_bytes(resp.content)
            decoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "framehash",
                    "-hash",
                    "sha256",
                    "-",
                ],
                text=True,
            )
            frames = [
                line.split(",")[-1].strip()
                for line in decoded.splitlines()
                if line and not line.startswith("#")
            ]
            row = dict(
                index=i,
                http_s=elapsed,
                mp4_sha256=digest(resp.content),
                rgb_sha256_by_frame=frames,
            )
            if (
                row["mp4_sha256"] != item["head_mp4"]
                or frames != item["rgb_sha256_by_frame"]
            ):
                raise ValueError(f"Delivered video differs at request {i}")
            records.append(row)
            (out / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f"{i}: {len(frames)} exact frames in {elapsed:.3f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "serve", "run"])
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--text-model-path", type=Path)
    p.add_argument("--out", type=Path, default=Path("camera-visit"))
    p.add_argument("--port", type=int, default=32960)
    p.add_argument("--control", default="preview-bcg")
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    model = a.model_path.resolve(strict=True)
    if a.action == "prepare":
        if not a.text_model_path:
            p.error("prepare needs --text-model-path")
        prepare(
            root / "model-view",
            model,
            a.text_model_path.resolve(strict=True),
            a.out.resolve(),
        )
        return
    spec = json.loads((root / "controls.json").read_text())[a.control]
    if a.action == "serve":
        args = list(spec["sglang_serve"])
        args[args.index("--model-path") + 1] = str(model)
        args[args.index("--port") + 1] = str(a.port)
        os.execvp("sglang", ["sglang", "serve", *args])
    generate(model, spec, a.out, a.port)


if __name__ == "__main__":
    main()
