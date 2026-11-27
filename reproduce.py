"""Reproduce one SGLang image-serving visit from the accompanying manifest."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time

import requests
from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "run"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--port", type=int, default=32150)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "workload.json").read_text())
    argv = list(manifest["sglang_serve"])
    model_path = str(args.model_path.resolve(strict=True))
    argv[argv.index("--model-path") + 1] = model_path
    argv[argv.index("--port") + 1] = str(args.port)
    if "--warmup-sampling-params" in argv:
        position = argv.index("--warmup-sampling-params") + 1
        params = json.loads(argv[position])
        if "image_path" in params:
            params["image_path"] = [str(root / p) for p in params["image_path"]]
        argv[position] = json.dumps(params)
    if args.action == "serve":
        env = {**os.environ, **manifest["server_environment"]}
        os.execvpe("sglang", ["sglang", "serve", *argv], env)

    url = f"http://127.0.0.1:{args.port}/v1/images"
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        for i, item in enumerate(manifest["requests"]):
            dest = args.out / f"{i:02d}"
            dest.mkdir()
            body = dict(
                item["body"], perf_dump_path=str((dest / "perf.json").resolve())
            )
            if body.get("model") == "{model-path}":
                body["model"] = model_path
            refs = []
            for spec in item["uploads"]:
                raw = (root / spec["file"]).read_bytes()
                if digest(raw) != spec["sha256"]:
                    raise ValueError(f"Reference hash mismatch for {spec['file']}")
                refs.append(("image", (spec["upload_name"], raw, "image/png")))
            if refs:
                form = {
                    k: str(v).lower() if isinstance(v, bool) else str(v)
                    for k, v in body.items()
                }
                start = time.perf_counter_ns()
                response = session.post(
                    url + "/edits", data=form, files=refs, timeout=3600
                )
            else:
                start = time.perf_counter_ns()
                response = session.post(url + "/generations", json=body, timeout=3600)
            end = time.perf_counter_ns()
            response.raise_for_status()
            (dest / "response.json").write_bytes(response.content)
            raw = base64.b64decode(
                response.json()["data"][0]["b64_json"], validate=True
            )
            (dest / "output.png").write_bytes(raw)
            saved = time.perf_counter_ns()
            with Image.open(io.BytesIO(raw)) as im:
                rgba_hash = digest(im.convert("RGBA").tobytes())
            perf = json.loads((dest / "perf.json").read_text())
            row = dict(
                index=i,
                role=item["role"],
                http_s=(end - start) / 1e9,
                saved_s=(saved - start) / 1e9,
                steps=len(perf["denoise_steps_ms"]),
                png_sha256=digest(raw),
                rgba_sha256=rgba_hash,
            )
            assert row["steps"] == item["steps"], row
            rows.append(row)
            (args.out / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    measured = [r["http_s"] for r in rows if r["role"] == "measure"]
    print(
        f"{len(measured)} timed requests, HTTP p50 {statistics.median(measured):.6f} s"
    )


if __name__ == "__main__":
    main()
