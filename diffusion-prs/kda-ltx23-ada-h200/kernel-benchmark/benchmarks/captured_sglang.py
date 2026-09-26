#!/usr/bin/env python3
"""Benchmark captured production shapes against pinned SGLang operators."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from bench_common import _emit_output_jsonl, summarize, time_fn
from captured_sglang_common import clone_result, poison_result

REPO_ROOT = Path(os.environ.get("BENCH_REPO_ROOT", Path(__file__).resolve().parents[1]))
CAPTURE_ROOT = REPO_ROOT / "captured_sglang"
_SGLANG_CONTEXT_OVERRIDE = None


@dataclass(frozen=True)
class PreparedCall:
    target: object
    args: tuple
    kwargs: dict

    def invoke(self):
        return self.target(*self.args, **self.kwargs)


def _load_module(path: Path, role: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    # Both roles: a baseline can sit next to shared helpers (the imported captures share
    # one adapter), and only the candidate's directory used to be importable.
    sys.path.insert(0, str(path.resolve().parent))
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"captured_sglang_{role}_{digest}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {role} module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_definition(value: str) -> Path:
    requested = Path(value)
    if requested.is_file():
        return requested.resolve()
    path = CAPTURE_ROOT / "definitions" / value
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _prepare(module, inputs: tuple) -> PreparedCall:
    prepare = getattr(module, "prepare_benchmark", None)
    if not callable(prepare):
        raise TypeError(f"{module.__name__} must expose prepare_benchmark(*inputs)")
    prepared = prepare(*inputs)
    if not isinstance(prepared, (tuple, list)) or len(prepared) != 3:
        raise TypeError("prepare_benchmark must return (callable, args, kwargs)")
    target, args, kwargs = prepared
    if (
        not callable(target)
        or not isinstance(args, (tuple, list))
        or not isinstance(kwargs, dict)
    ):
        raise TypeError("invalid prepared benchmark call")
    return PreparedCall(target, tuple(args), kwargs)


def _run(module, inputs: tuple):
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"{module.__name__} must expose run(*inputs)")
    return run(*inputs)


def _flatten_result(result) -> tuple:
    return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def _check_output_aliases(baseline, result, inputs: tuple) -> tuple[bool, str]:
    check = getattr(baseline, "check_output_aliases", None)
    if callable(check):
        ok, note = check(result, inputs)
        if not ok:
            return False, note
    output_tensors = getattr(baseline, "output_tensors", None)
    if not callable(output_tensors):
        raise TypeError(f"{baseline.__name__} must expose output_tensors(inputs)")
    expected = tuple(output_tensors(inputs))
    if not expected:
        return True, "not required"
    actual = _flatten_result(result)
    if len(actual) != len(expected):
        return False, "wrong number of returned outputs"
    for index, (got, out) in enumerate(zip(actual, expected, strict=True)):
        if not isinstance(got, torch.Tensor) or not isinstance(out, torch.Tensor):
            return False, f"output {index} is not a tensor"
        if (
            got.data_ptr() != out.data_ptr()
            or got.shape != out.shape
            or got.stride() != out.stride()
            or got.storage_offset() != out.storage_offset()
        ):
            return False, f"output {index} does not alias the caller-owned destination"
    return True, "ok"


def _check_input_state(baseline, actual_inputs: tuple, expected_inputs: tuple):
    compare = getattr(baseline, "compare_input_state", None)
    if callable(compare):
        return compare(actual_inputs, expected_inputs)
    return True, "not required"


def _seed(definition: str, workload_id: str) -> int:
    material = f"{definition}\0{workload_id}\0{os.environ.get('BENCH_INPUT_SALT', '')}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")


def _make_inputs(baseline, row: dict, device: torch.device, seed: int) -> tuple:
    return tuple(baseline.make_inputs(row, device=device, seed=seed))


def _poison_owned_outputs(baseline, inputs: tuple) -> None:
    poison = getattr(baseline, "poison_outputs", None)
    if callable(poison):
        poison(inputs)
        return
    output_tensors = getattr(baseline, "output_tensors", None)
    if not callable(output_tensors):
        raise TypeError(f"{baseline.__name__} must expose output_tensors(inputs)")
    poison_result(tuple(output_tensors(inputs)))


def _select_workloads(rows: list[dict], baseline) -> list[dict]:
    fast_mix = int(os.environ.get("BENCH_FAST_MIX", "0"))
    max_rows = int(os.environ.get("BENCH_MAX_OFFICIAL", "0"))
    if fast_mix > 0:
        small = [row for row in rows if baseline.size_class(row) == "small"]
        large = [row for row in rows if baseline.size_class(row) == "large"]
        selected = small[: (fast_mix + 1) // 2] + large[: fast_mix // 2]
        if len(selected) < fast_mix:
            selected += [row for row in rows if row not in selected][
                : fast_mix - len(selected)
            ]
        selected_ids = {id(row) for row in selected}
        return [row for row in rows if id(row) in selected_ids]
    return rows[:max_rows] if max_rows > 0 else rows


def _check_gpu(definition: dict, device: torch.device) -> None:
    required = str(definition["target_gpu"])
    actual = torch.cuda.get_device_name(device)
    required_token = required.split()[-1]
    if required_token not in actual:
        raise RuntimeError(f"task requires {required}, gpu-run provided {actual}")


def _check_sglang_runtime(definition: dict) -> str:
    global _SGLANG_CONTEXT_OVERRIDE
    baseline = definition.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("kind") != "sglang_runtime":
        raise RuntimeError(
            "captured task must declare a pinned sglang_runtime baseline"
        )
    actual = importlib.metadata.version("sglang")
    expected = baseline.get("sglang_version")
    # A task captured on a released SGLang pins the wheel version. A task
    # captured on a fork branch (a PR stack that never got a version bump)
    # cannot: every commit on it reports the same dev version, so the pin that
    # actually distinguishes the tree is ``sglang_revision``, asserted when the
    # runtime image is built. Requiring a version here would reject the very
    # image the definition names.
    if expected is not None and actual != str(expected):
        raise RuntimeError(
            f"task requires SGLang {expected}, runtime provides {actual}"
        )
    if expected is None and not baseline.get("sglang_revision"):
        raise RuntimeError("captured task must pin sglang_version or sglang_revision")
    # Standalone operator replay does not construct an SGLang server, but a few
    # production kernels read immutable execution flags from RuntimeContext.
    # Publish the same default config boundary before importing task modules.
    from sglang.srt.runtime_context import get_context, get_exec

    try:
        get_exec()
    except ValueError:
        _SGLANG_CONTEXT_OVERRIDE = get_context().override_server_args(
            moe_runner_backend="triton"
        )
        _SGLANG_CONTEXT_OVERRIDE.install()
    return actual


def _correctness(baseline, candidate, row: dict, device: torch.device, seed: int):
    baseline_inputs = _make_inputs(baseline, row, device, seed)
    candidate_inputs = _make_inputs(baseline, row, device, seed)
    expected = clone_result(_run(baseline, baseline_inputs))
    _poison_owned_outputs(baseline, candidate_inputs)
    actual_alias = _run(candidate, candidate_inputs)
    aliases_ok, alias_note = _check_output_aliases(
        baseline, actual_alias, candidate_inputs
    )
    if not aliases_ok:
        return False, "incorrect", f"plain run: {alias_note}"
    state_ok, state_note = _check_input_state(
        baseline, candidate_inputs, baseline_inputs
    )
    if not state_ok:
        return False, "incorrect", f"plain run input state: {state_note}"
    actual = clone_result(actual_alias)
    ok, note = baseline.compare(actual, expected, row)
    if not ok:
        return False, "incorrect", f"plain run: {note}"

    prepared_inputs = _make_inputs(baseline, row, device, seed)
    prepared = _prepare(candidate, prepared_inputs)
    _poison_owned_outputs(baseline, prepared_inputs)
    prepared_actual_alias = prepared.invoke()
    aliases_ok, alias_note = _check_output_aliases(
        baseline, prepared_actual_alias, prepared_inputs
    )
    if not aliases_ok:
        return False, "incorrect", f"prepared run: {alias_note}"
    state_ok, state_note = _check_input_state(
        baseline, prepared_inputs, baseline_inputs
    )
    if not state_ok:
        return False, "incorrect", f"prepared run input state: {state_note}"
    prepared_actual = clone_result(prepared_actual_alias)
    ok, note = baseline.compare(prepared_actual, expected, row)
    if not ok:
        return False, "incorrect", f"prepared run: {note}"

    mutate = getattr(baseline, "mutate_inputs", None)
    if callable(mutate):
        live_baseline_inputs = _make_inputs(baseline, row, device, seed)
        live_candidate_inputs = _make_inputs(baseline, row, device, seed)
        live_prepared = _prepare(candidate, live_candidate_inputs)
        # Perturb after preparation, then reuse the same callable with another
        # perturbation. This rejects both eager prepare-time computation and
        # lazy caching of the first invocation. For read/write input aliases,
        # advance reference and candidate exactly once per iteration so their
        # next mutations start from the same state.
        for replay in range(2):
            mutate(live_baseline_inputs)
            mutate(live_candidate_inputs)
            live_expected = clone_result(_run(baseline, live_baseline_inputs))
            _poison_owned_outputs(baseline, live_candidate_inputs)
            live_actual_alias = live_prepared.invoke()
            aliases_ok, alias_note = _check_output_aliases(
                baseline, live_actual_alias, live_candidate_inputs
            )
            if not aliases_ok:
                return (
                    False,
                    "incorrect",
                    f"prepared input liveness {replay + 1}: {alias_note}",
                )
            state_ok, state_note = _check_input_state(
                baseline, live_candidate_inputs, live_baseline_inputs
            )
            if not state_ok:
                return (
                    False,
                    "incorrect",
                    f"prepared input liveness {replay + 1} input state: {state_note}",
                )
            live_actual = clone_result(live_actual_alias)
            ok, note = baseline.compare(live_actual, live_expected, row)
            if not ok:
                return False, "incorrect", f"prepared input liveness {replay + 1}: {note}"
    return True, "ok", "ok"


def _time_pair(
    baseline,
    candidate,
    baseline_inputs: tuple,
    candidate_inputs: tuple,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    trials: int,
) -> tuple[float, float, str]:
    baseline_call = _prepare(baseline, baseline_inputs)
    candidate_call = _prepare(candidate, candidate_inputs)
    baseline_call.invoke()
    candidate_call.invoke()
    torch.cuda.synchronize(device)

    baseline_trials = []
    candidate_trials = []
    timer = None
    for trial in range(trials):
        order = (
            ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        )
        measured = {}
        for role in order:
            call = baseline_call if role == "baseline" else candidate_call
            value, current_timer = time_fn(call.invoke, (), device, warmup, iters)
            timer = timer or current_timer
            if current_timer != timer:
                raise RuntimeError(f"timer mismatch: {current_timer} != {timer}")
            measured[role] = value
        baseline_trials.append(measured["baseline"])
        candidate_trials.append(measured["candidate"])
    return (
        statistics.median(baseline_trials),
        statistics.median(candidate_trials),
        str(timer),
    )


def run_definition(
    definition: dict,
    candidate_path: Path,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    trials: int,
) -> list[dict]:
    name = definition["name"]
    sglang_version = _check_sglang_runtime(definition)
    baseline = _load_module(CAPTURE_ROOT / "baselines" / f"{name}.py", "baseline")
    candidate = _load_module(candidate_path.resolve(), "candidate")
    workload_path = CAPTURE_ROOT / "workloads" / f"{name}.json"
    workloads = _select_workloads(json.loads(workload_path.read_text()), baseline)
    if not workloads:
        raise ValueError(f"no workloads in {workload_path}")

    print(f"OP: {name}")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"sglang: {sglang_version}")
    print(f"workloads: {len(workloads)}")
    print(f"warmup/iters/trials: {warmup}/{iters}/{trials}")
    print()

    rows = []
    for index, workload in enumerate(workloads, start=1):
        workload_id = str(workload["id"])
        axes = baseline.axes(workload)
        print(f"[{index:03d}/{len(workloads):03d}] {workload_id} axes={axes}")
        passed = False
        status = "error"
        note = ""
        baseline_ms = None
        kernel_ms = None
        speedup = None
        try:
            seed = _seed(name, workload_id)
            passed, status, note = _correctness(
                baseline, candidate, workload, device, seed
            )
            if passed:
                baseline_inputs = _make_inputs(baseline, workload, device, seed)
                candidate_inputs = _make_inputs(baseline, workload, device, seed)
                baseline_ms, kernel_ms, timer = _time_pair(
                    baseline,
                    candidate,
                    baseline_inputs,
                    candidate_inputs,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                    trials=trials,
                )
                speedup = baseline_ms / kernel_ms
                print(f"  baseline: {baseline_ms:.6f} ms")
                print(f"  kernel:   {kernel_ms:.6f} ms")
                print(f"  speedup:  {speedup:.4f}x ({timer})")
                print("  correct:  PASS")
            else:
                print(f"  correct:  FAIL {note}")
        except Exception as exc:  # noqa: BLE001 - report task failures per row.
            passed = False
            status = "error"
            note = f"{type(exc).__name__}: {exc}"
            print(f"  correct:  ERROR {note}")
        print()
        rows.append(
            {
                "suite": "production",
                "id": workload_id,
                "axes": axes,
                "size_class": baseline.size_class(workload),
                "passed": passed,
                "status": status,
                "note": note,
                "baseline_ms": baseline_ms,
                "kernel_ms": kernel_ms,
                "speedup": speedup,
            }
        )

    summarize(rows, "")
    _emit_output_jsonl(name, rows)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--trials", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    definition = json.loads(_resolve_definition(args.definition).read_text())
    _check_gpu(definition, device)
    timing = definition["timing"]
    rows = run_definition(
        definition,
        args.candidate,
        device=device,
        warmup=args.warmup or int(os.environ.get("BENCH_WARMUP", timing["warmup"])),
        iters=args.iters or int(os.environ.get("BENCH_ITERS", timing["iterations"])),
        trials=args.trials or int(os.environ.get("BENCH_TRIALS", timing["trials"])),
    )
    return (
        1
        if os.environ.get("BENCH_FAIL_ON_ERROR") == "1"
        and any(not row["passed"] for row in rows)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
