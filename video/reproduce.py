"""Prepare the measured config view and replay the three LTX image-to-video requests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import requests


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "serve", "run"))
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("ltx-visit"))
    p.add_argument("--port", type=int, default=32150)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    m = json.loads((root / "workload.json").read_text())
    model = a.model_path.resolve(strict=True)
    if a.action == "prepare":
        target = a.out.resolve()
        if target.is_relative_to(model):
            raise ValueError("Choose a sibling directory for the native view")
        target.mkdir(exist_ok=False, parents=True)
        # The native LTX-2.3 registry recognizes the model directory's basename.
        target = target / "LTX-2.3-Diffusers"
        target.mkdir()
        aliases = {x["file"]: x for x in m["metadata_view"]}
        for file in model.rglob("*"):
            rel = file.relative_to(model)
            name = rel.as_posix()
            if ".cache" in rel.parts or not file.is_file():
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if name in aliases:
                data = (root / "config-aliases" / rel).read_bytes()
                item = aliases[name]
                if (
                    digest(data) != item["measured_config_sha256"]
                    or digest(file.read_bytes()) != item["canonical_sha256"]
                ):
                    raise ValueError(name)
                dst.write_bytes(data)
            else:
                dst.symlink_to(file)
        for name in aliases:
            if not (target / name).is_file():
                raise ValueError(name)
        return
    if a.action == "serve":
        prepared = model / "LTX-2.3-Diffusers"
        if (prepared / "model_index.json").is_file():
            model = prepared
        argv = list(m["sglang_serve"])
        argv[argv.index("--model-path") + 1] = str(model)
        argv[argv.index("--port") + 1] = str(a.port)
        if m.get("torch_cudnn_deterministic"):
            os.environ["PYTHONPATH"] = str(root / "torch-settings") + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.execvp("sglang", ["sglang", "serve", *argv])
    a.out.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{a.port}/v1/videos"
    with requests.Session() as http:
        http.trust_env = False
        for i, spec in enumerate(m["requests"]):
            ref = root / spec["reference"]
            image = ref.read_bytes()
            if digest(image) != spec["reference_sha256"]:
                raise ValueError(ref.name)
            dest = (a.out / f"{i:02d}").resolve()
            dest.mkdir()
            form = dict(
                spec["body"],
                output_path=str(dest / "native-output"),
                perf_dump_path=str(dest / "perf.json"),
            )
            form = {
                k: (v if isinstance(v, str) else json.dumps(v)) for k, v in form.items()
            }
            resp = http.post(
                base,
                data=form,
                files={"input_reference": (ref.name, image, "image/png")},
                timeout=600,
            )
            resp.raise_for_status()
            job = resp.json()
            url = base + "/" + job["id"]
            deadline = time.monotonic() + 1800
            while job["status"] not in ("completed", "failed", "cancelled", "canceled"):
                if time.monotonic() > deadline:
                    raise TimeoutError(job["id"])
                time.sleep(0.1)
                resp = http.get(url, timeout=90)
                resp.raise_for_status()
                job = resp.json()
            if job["status"] != "completed":
                raise RuntimeError(job)
            resp = http.get(url + "/content", timeout=240)
            resp.raise_for_status()
            path = dest / "output.mp4"
            path.write_bytes(resp.content)
            framehash = subprocess.check_output(
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
                x.split(",")[-1].strip()
                for x in framehash.splitlines()
                if x and not x.startswith("#")
            ]
            pcm = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "-",
                ]
            )
            verified = dict(
                mp4=digest(resp.content) == spec["head_mp4"],
                rgb=frames == spec["rgb_sha256_by_frame"],
                pcm=digest(pcm) == spec["audio"]["f32le_sha256"],
            )
            print(
                json.dumps(dict(index=i, frames=len(frames), equal=verified)),
                flush=True,
            )
            if not all(verified.values()):
                raise RuntimeError(f"Video or audio differs at request {i}")


if __name__ == "__main__":
    main()
