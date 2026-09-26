"""Compare pinned baseline and integrated Kimi winner on full native layouts."""

import gc
import fcntl
import os
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/hunyuan-full-native-kernel-comparison"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hunyuan_qkv_rope_pack


def materialize(meta):
    shape, stride = meta["shape"], meta["stride"]
    offset = meta.get("storage_offset", 0)
    elements = offset + 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
    dtype = getattr(torch, meta["dtype"].split(".")[-1])
    storage = torch.randn(elements, device="cuda", dtype=dtype)
    return storage.as_strided(shape, stride, offset)


def main():
    lock = Path('/tmp/kda-gpu-locks/gpu-4.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    OUT.mkdir(exist_ok=False)
    paths = {
        'baseline': ROOT / 'baseline/python/sglang/kernels/ops/diffusion/rope/hunyuan_qkv_pack_triton.py',
        'candidate': ROOT / 'pr-hunyuan-qkv/python/sglang/kernels/ops/diffusion/rope/hunyuan_qkv_pack_triton.py',
    }
    functions = {name: load(name, path) for name, path in paths.items()}
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    capture = ROOT / "validation/hunyuan-qkv-full65f-native-only.json"
    rows = [
        row
        for row in json.loads(capture.read_text())
        if row["function"].endswith(".hunyuan_qkv_rope_pack")
    ]
    result = []
    for index, row in enumerate(rows):
        torch.manual_seed(8300 + index)
        args = [materialize(meta) for meta in row["args"]]
        before = [arg.clone() for arg in args]
        # Both original and changed inputs must match exact baseline bytes.
        for replay in range(2):
            if replay:
                for x in args:
                    x.add_(0.125)
            reference = functions["baseline"](*args)
            for name in ("candidate",):
                actual = functions[name](*args)
                for expected, value, meta in zip(reference, actual, row["result"]):
                    assert torch.equal(
                        expected.view(torch.uint8), value.view(torch.uint8)
                    ), (index, name, replay)
                    assert (
                        list(value.shape) == meta["shape"]
                        and list(value.stride()) == meta["stride"]
                    )
                    assert value.storage_offset() == meta["storage_offset"]
                    assert all(
                        value.untyped_storage().data_ptr()
                        != x.untyped_storage().data_ptr()
                        for x in args
                    )
                assert len({x.untyped_storage().data_ptr() for x in actual}) == 1
                del actual
            del reference
        assert all(
            torch.equal(x, (old + 0.125).to(old.dtype)) for x, old in zip(args, before)
        )
        del before
        repetitions = []
        for repeat in range(3):
            order = list(functions)
            if repeat % 2:
                order.reverse()
            times = {}
            for name in order:
                times[name] = triton.testing.do_bench_cudagraph(
                    lambda: functions[name](*args), rep=200
                )
            repetitions.append(times)
        record = {
            "capture_row": index,
            "image_tokens": args[0].shape[1],
            "text_tokens": args[3].shape[1],
            "image_v_stride": list(args[2].stride()),
            "text_v_offset": args[5].storage_offset(),
            "call_count": row.get("call_count"),
            "correctness": "byte-exact two input states; layout/offset/alias/input state checked",
            "milliseconds": repetitions,
        }
        result.append(record)
        print(json.dumps(record), flush=True)
        del args
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        name: [
            math.exp(
                sum(
                    math.log(
                        r["milliseconds"][repeat]["baseline"]
                        / r["milliseconds"][repeat][name]
                    )
                    for r in result
                )
                / len(result)
            )
            for repeat in range(3)
        ]
        for name in ("candidate",)
    }
    manifest = {
        "source_sha256": source_hashes,
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "actual_values": "random regenerated; exact captured shape/dtype/stride/storage_offset",
        "rows": result,
        "geomean_speedups": summary,
        "timing": "triton.testing.do_bench_cudagraph rep=200ms, three repeats alternating candidate order; no E2E claim",
        "integration_compatibility_change": "restore baseline explicit FP32 conversion immediately after cos/sin loads",
    }
    (OUT / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
