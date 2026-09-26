#!/usr/bin/env python3
"""Small helpers for standalone AKO kernel probes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import size_policy
import torch

# The packaged CuTe-DSL baselines (e.g. GDN prefill) use the enum-form
# cute.arch.ProxyKind / SharedSpace, which neither the 4.5.2 nor the 4.6.0 cu13
# wheel exports natively. Install the compatibility aliases once here — every real
# bench run (the judge, run_baselines) imports bench_common, so a
# CuTe baseline always has them before it compiles. Guarded so CPU-only / non-CuTe
# environments (no cutlass installed) still import bench_common.
try:
    from cutlass_compat import install_cutlass_452_cu13_compat as _install_cute_compat

    _install_cute_compat()
except Exception:
    pass


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("BENCH_REPO_ROOT", WORKSPACE_ROOT))
TRACE_ROOT = Path(os.environ.get("BENCH_TRACE_ROOT", REPO_ROOT / "flashinfer_trace"))
CONTEST_ROOT = TRACE_ROOT
CANDIDATE_ROOT = Path(
    os.environ.get("BENCH_CANDIDATE_ROOT", REPO_ROOT / "candidates" / "ako")
)

TIMER_PROTOCOL = {
    "implementation": "flashinfer.testing.bench_gpu_time_with_cupti",
    "required_cupti_major": 13,
    "cold_l2_cache": True,
    "use_cuda_graph": False,
    "within_trial_aggregation": "median",
    "across_trial_aggregation": "median",
}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw in (None, "") else raw


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return default if raw in (None, "") else Path(raw).expanduser()


def choose_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        device = torch.device(device_setting)
        torch.cuda.set_device(device)
        return device

    rows = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    candidates = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = None
    if visible:
        visible_ids = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    for row in rows:
        idx_s, mem_s, util_s = [part.strip() for part in row.split(",")]
        if visible_ids is not None and int(idx_s) not in visible_ids:
            continue
        candidates.append((int(mem_s), int(util_s), int(idx_s)))
    mem, util, idx = min(candidates)
    local_idx = visible_ids.index(idx) if visible_ids is not None else idx
    device = torch.device(f"cuda:{local_idx}")
    torch.cuda.set_device(device)
    print(f"selected GPU cuda:{idx} (memory_used={mem} MiB, util={util}%)")
    return device


def load_json(path: Path):
    return json.loads(path.read_text())


def load_workloads(path: Path, max_workloads: int):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if max_workloads > 0:
        rows = rows[:max_workloads]
    return rows


def _entry_from_row(row):
    wl = row["workload"]
    generated = bool(row.get("generated"))
    axes = wl["axes"]
    uuid = wl.get("uuid", "")
    wid = uuid[:8] if uuid else "-".join(f"{k}{axes[k]}" for k in sorted(axes))
    return {
        "suite": "large" if generated else "official",
        "id": wid,
        "axes": axes,
        "raw": None if generated else wl,
    }


def _row_size_class(row):
    sc = row.get("size_class")
    if sc in size_policy.VALID_SIZE_CLASSES:
        return sc
    return size_policy.classify(row.get("definition", ""), row["workload"]["axes"])


def _fast_size_mix(rows, count):
    """Pick ``count`` workloads balanced across ``size_class`` for the dev subset.

    Drawn from the WHOLE pool (generated included) so a kernel whose large shapes
    are all generated (e.g. DSA) still gets large coverage; small takes the odd
    extra since it is the cheaper regime. Deterministic: file order within each
    class, and a class that is short backfills from the other.
    """
    small = [r for r in rows if _row_size_class(r) == size_policy.SMALL]
    large = [r for r in rows if _row_size_class(r) == size_policy.LARGE]
    n_small = (count + 1) // 2
    n_large = count // 2
    picked = small[:n_small] + large[:n_large]
    if len(picked) < count:
        picked += (small[n_small:] + large[n_large:])[: count - len(picked)]
    chosen = {id(r) for r in picked}
    return [_entry_from_row(r) for r in rows if id(r) in chosen]


def make_entries(
    rows, *, include_official=True, include_large=True, max_official=0, fast_mix=0
):
    """Build run_benchmark entries from dataset rows.

    ``suite`` preserves its original meaning: a generated row (``generated:
    true``, formerly a runtime-only large shape) has no input blobs, so its entry
    is ``suite="large"`` with ``raw=None`` and make_inputs produces the inputs
    deterministically at run time; every other dataset row is ``suite="official"``
    and keeps ``raw`` set so make_inputs loads its safetensors.
    ``include_large`` gates generated rows,
    ``include_official`` gates the rest, and ``max_official`` caps only the
    non-generated rows (a smoke-test knob). The small/large size split used for
    reporting is orthogonal and comes from ``size_policy.classify`` at run time.

    ``fast_mix`` (when > 0) overrides the gating knobs and returns a small dev
    subset balanced across ``size_class`` (see ``_fast_size_mix``), so the quick
    iteration set always exercises both a small and a large shape.
    """
    if fast_mix > 0:
        return _fast_size_mix(rows, fast_mix)
    entries = []
    official_used = 0
    for row in rows:
        generated = bool(row.get("generated"))
        if generated:
            if not include_large:
                continue
        else:
            if not include_official:
                continue
            if max_official > 0 and official_used >= max_official:
                continue
            official_used += 1
        entries.append(_entry_from_row(row))
    return entries


class EnvironmentMismatch(RuntimeError):
    """Raised when the captured benchmark environment is incomplete or invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _module_tree_sha256(module_name: str) -> str:
    """Hash installed code/data that can affect a benchmark implementation."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return "missing"
    roots = [Path(path) for path in spec.submodule_search_locations or []]
    if not roots and spec.origin:
        roots = [Path(spec.origin)]
    files = []
    for root in roots:
        if root.is_file():
            files.append((root.name, root))
            continue
        files.extend(
            (str(path.relative_to(root)), path)
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    return _sha256_json(
        [
            (name, _sha256_file(path))
            for name, path in sorted(files, key=lambda row: row[0])
        ]
    )


def _python_packages_sha256() -> str:
    """Fingerprint the exact installed Python distribution inventory."""
    rows = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        direct_url = distribution.read_text("direct_url.json") or ""
        rows.append((name, distribution.version, direct_url))
    return _sha256_json(sorted(rows))


def _cupti_python_version() -> str:
    try:
        return importlib.metadata.version("cupti-python")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _flashinfer_identity() -> dict:
    """Identify the exact FlashInfer MoE implementation and cubin manifest."""
    try:
        flashinfer = importlib.import_module("flashinfer")
        version = str(flashinfer.__version__)
        artifacts = importlib.import_module("flashinfer.artifacts")
        bmm_path = str(artifacts.ArtifactPath.TRTLLM_GEN_BMM)
        bmm_checksum = str(artifacts.CheckSumHash.TRTLLM_GEN_BMM)
    except (
        AttributeError,
        ImportError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        raise EnvironmentMismatch(
            "cannot attest the FlashInfer TRT-LLM BMM implementation"
        ) from exc
    if not version or not bmm_path or not bmm_checksum:
        raise EnvironmentMismatch(
            "cannot attest the FlashInfer TRT-LLM BMM implementation"
        )
    return {
        "python_version": version,
        "trtllm_gen_bmm_path": bmm_path,
        "trtllm_gen_bmm_manifest_sha256": bmm_checksum,
    }


def _require_cupti() -> str:
    """Fail before FlashInfer can take its internal CUDA-event fallback."""
    try:
        importlib.import_module("cupti.cupti")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "CUPTI Python bindings are unavailable; timer fallback is forbidden"
        ) from exc
    version = _cupti_python_version()
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"invalid cupti-python version {version!r}") from exc
    required = TIMER_PROTOCOL["required_cupti_major"]
    if major < required:
        raise RuntimeError(
            f"cupti-python {version} is too old; CUPTI {required}+ is required and "
            "timer fallback is forbidden"
        )
    return version


def _nvcc_version() -> str:
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"
    return " ".join(result.stdout.split())


def _decode_nvml(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _physical_device_token(device: torch.device) -> str:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(index)
    tokens = [token.strip() for token in visible.split(",")]
    if index >= len(tokens) or not tokens[index]:
        raise EnvironmentMismatch(
            f"cannot map {device} through CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return tokens[index]


def _nvml_device_environment(device: torch.device) -> dict:
    """Capture device identity, configuration, and runtime state for audit."""
    try:
        import pynvml
    except Exception as exc:
        raise EnvironmentMismatch(
            "pynvml is required to identify the physical GPU"
        ) from exc

    pynvml.nvmlInit()
    try:
        token = _physical_device_token(device)
        if token.startswith(("GPU-", "MIG-")):
            handle = pynvml.nvmlDeviceGetHandleByUUID(token)
        elif token.isdigit():
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(token))
        else:
            raise EnvironmentMismatch(
                f"unsupported CUDA_VISIBLE_DEVICES token {token!r}"
            )

        def optional(call, *args):
            try:
                return call(handle, *args)
            except pynvml.NVMLError:
                return "unsupported"

        pci = optional(pynvml.nvmlDeviceGetPciInfo)
        ecc = optional(pynvml.nvmlDeviceGetEccMode)
        mig = optional(pynvml.nvmlDeviceGetMigMode)
        gpu_uuid = _decode_nvml(pynvml.nvmlDeviceGetUUID(handle))
        if not gpu_uuid.startswith(("GPU-", "MIG-")):
            raise EnvironmentMismatch(f"NVML returned invalid GPU UUID {gpu_uuid!r}")
        return {
            "gpu_uuid": gpu_uuid,
            "nvml_gpu_name": _decode_nvml(pynvml.nvmlDeviceGetName(handle)),
            "pci_bus_id": (
                _decode_nvml(pci.busId) if pci != "unsupported" else "unsupported"
            ),
            "driver_version": _decode_nvml(pynvml.nvmlSystemGetDriverVersion()),
            "power_limit_mw": optional(pynvml.nvmlDeviceGetPowerManagementLimit),
            "max_sm_clock_mhz": optional(
                pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_SM
            ),
            "max_memory_clock_mhz": optional(
                pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_MEM
            ),
            "current_sm_clock_mhz": optional(
                pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM
            ),
            "current_memory_clock_mhz": optional(
                pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_MEM
            ),
            "performance_state": optional(pynvml.nvmlDeviceGetPerformanceState),
            "compute_mode": optional(pynvml.nvmlDeviceGetComputeMode),
            "ecc_mode_current_pending": list(ecc)
            if isinstance(ecc, (list, tuple))
            else ecc,
            "mig_mode_current_pending": list(mig)
            if isinstance(mig, (list, tuple))
            else mig,
        }
    finally:
        pynvml.nvmlShutdown()


def _cute_dsl_version() -> str:
    """The pinned CuTe-DSL (``nvidia-cutlass-dsl``) version.

    Recorded in the benchmark environment: a CuTe-DSL kernel's codegen — and thus
    its latency — can change across CuTe-DSL releases (e.g. the 4.5 -> 4.6 bump
    that adds IKET and changes some ``cute.arch`` spellings). Returns ``"unknown"``
    off the pinned environment.
    """
    import importlib.metadata as _md

    for dist in ("nvidia-cutlass-dsl", "nvidia_cutlass_dsl"):
        try:
            return _md.version(dist)
        except Exception:
            pass
    try:
        import cutlass

        return str(getattr(cutlass, "__version__", "unknown"))
    except Exception:
        return "unknown"


def capture_environment(device: torch.device) -> dict:
    """Capture the exact hardware and software identity used by a frozen trace."""
    props = torch.cuda.get_device_properties(device)
    env = {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "multiprocessor_count": props.multi_processor_count,
        "l2_cache_size_bytes": getattr(props, "L2_cache_size", "unavailable"),
        "nvcc_version": _nvcc_version(),
        "cute_dsl_version": _cute_dsl_version(),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_version": str(torch.__version__),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "libc": list(platform.libc_ver()),
        "python_packages_sha256": _python_packages_sha256(),
        "flashinfer_tree_sha256": _module_tree_sha256("flashinfer"),
        "flashinfer_identity": _flashinfer_identity(),
        "cupti_python_version": _cupti_python_version(),
        "runtime_environment": {
            name: os.environ.get(name, "")
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "CUDA_MODULE_LOADING",
                "CUDA_LAUNCH_BLOCKING",
                "CUDA_CACHE_DISABLE",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
    }
    env.update(_nvml_device_environment(device))
    unattested = [
        key
        for key, value in env.items()
        if isinstance(value, str)
        and (
            value in {"missing", "unknown", "unavailable", "unsupported"}
            or value.startswith("unavailable:")
        )
    ]
    if unattested:
        raise EnvironmentMismatch(
            f"cannot attest the complete frozen-baseline environment: {unattested}"
        )
    return env


def resolve_tensor_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return TRACE_ROOT / p


def load_safetensor(spec: dict, device: torch.device) -> torch.Tensor:
    import safetensors.torch as st

    path = resolve_tensor_path(spec["path"])
    tensor = st.load_file(str(path))[spec["tensor_key"]].contiguous()
    return tensor.to(device=device, non_blocking=True)


def load_kernel(path: Path, module_name: str = "candidate_kernel"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run


def load_kernel_source(
    source: str, filename: str, module_name: str = "candidate_kernel"
):
    module = types.ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module.run


def load_kernel_git_blob(
    repo: Path, blob_path: str, module_name: str = "candidate_kernel"
):
    source = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"HEAD:{blob_path}"],
        text=True,
    )
    # Triton @jit functions require inspectable real files. Cache recovered
    # git blobs locally rather than restoring deleted files outside this workdir.
    digest = hashlib.sha256(f"{repo}:{blob_path}\n{source}".encode()).hexdigest()[:16]
    generated = WORKSPACE_ROOT / "artifacts" / "generated_git_blobs"
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"{Path(blob_path).stem}_{digest}.py"
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    return load_kernel(path, module_name)


def load_solution_json(path: Path, package_name: str):
    data = load_json(path)
    root = Path(tempfile.mkdtemp(prefix=f"{package_name}_"))
    package_root = root / package_name
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")

    for source in data["sources"]:
        dst = package_root / source["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(source["content"])
        cur = dst.parent
        while cur != package_root.parent:
            init = cur / "__init__.py"
            if not init.exists():
                init.write_text("")
            if cur == package_root:
                break
            cur = cur.parent

    sys.path.insert(0, str(root))
    module = importlib.import_module(f"{package_name}.main")
    return module.run


def rand_tensor(shape, dtype, device, *, positive: bool = False):
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        if positive:
            return (
                torch.rand(shape, dtype=torch.float32, device=device) * 0.1 + 0.01
            ).to(dtype)
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype is torch.float8_e4m3fn:
        return (
            torch.randn(shape, dtype=torch.float32, device=device)
            .clamp_(-2.0, 2.0)
            .to(dtype)
        )
    if dtype is torch.int32:
        return torch.randint(0, 1024, shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported dtype: {dtype}")


def normalize_outputs(result):
    if result is None:
        return []
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    return [torch.as_tensor(result)]


def clone_arg(arg):
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    if isinstance(arg, tuple):
        return tuple(clone_arg(x) for x in arg)
    if isinstance(arg, list):
        return [clone_arg(x) for x in arg]
    if isinstance(arg, dict):
        return {k: clone_arg(v) for k, v in arg.items()}
    return arg


def clone_args(args):
    return tuple(clone_arg(arg) for arg in args)


def time_fn(fn, args, device, warmup: int, iters: int):
    del device  # Kept in the shared timing API; CUPTI selects the active CUDA device.
    _require_cupti()
    flashinfer_testing = importlib.import_module("flashinfer.testing")

    def reject_fallback(*_args, **_kwargs):
        raise RuntimeError("FlashInfer attempted a non-CUPTI timer fallback")

    fallback_names = (
        "bench_gpu_time_with_cuda_event",
        "bench_gpu_time_with_cudagraph",
    )
    fallback_timers = {
        name: getattr(flashinfer_testing, name)
        for name in fallback_names
        if hasattr(flashinfer_testing, name)
    }
    try:
        for name in fallback_timers:
            setattr(flashinfer_testing, name, reject_fallback)
        times = [
            float(value)
            for value in flashinfer_testing.bench_gpu_time_with_cupti(
                fn=fn,
                dry_run_iters=warmup,
                repeat_iters=iters,
                input_args=tuple(args),
                cold_l2_cache=TIMER_PROTOCOL["cold_l2_cache"],
                use_cuda_graph=TIMER_PROTOCOL["use_cuda_graph"],
            )
        ]
    finally:
        for name, timer in fallback_timers.items():
            setattr(flashinfer_testing, name, timer)
    if not times or any(not math.isfinite(value) or value <= 0 for value in times):
        raise RuntimeError(f"CUPTI returned invalid timings: {times!r}")
    return statistics.median(times), "cupti"


# Notes returned by compare_outputs when the candidate fails the correctness gate
# (as opposed to raising a runtime exception). Used to label a failed workload.
_CORRECTNESS_FAIL_NOTES = frozenset(
    {
        "degenerate output",
        "numerical mismatch",
        "non-finite output",
        "wrong shape",
        "wrong number of outputs",
    }
)


# Loose elementwise tolerances can accept zeros/constants when references are small.
_MAX_REL_L2 = 0.25


def compare_outputs(
    candidate, baseline, atol: float, rtol: float, required_matched_ratio
):
    cand = normalize_outputs(candidate)
    base = normalize_outputs(baseline)
    if len(cand) < len(base):
        return False, float("inf"), float("inf"), 0.0, "wrong number of outputs"

    max_abs = 0.0
    max_rel = 0.0
    total = 0
    bad = 0
    for got, ref in zip(cand[: len(base)], base, strict=True):
        if tuple(got.shape) != tuple(ref.shape):
            return False, float("inf"), float("inf"), 0.0, "wrong shape"
        if not torch.isfinite(got.float()).all().item():
            return False, float("inf"), float("inf"), 0.0, "non-finite output"
        x = got.float()
        y = ref.float()
        ref_norm = float(torch.linalg.vector_norm(y).item())
        if ref_norm > 0:
            rel_l2 = float(torch.linalg.vector_norm(x - y).item()) / ref_norm
            if rel_l2 > _MAX_REL_L2:
                return False, float("inf"), float("inf"), 0.0, "degenerate output"
        abs_error = torch.abs(x - y)
        rel_error = abs_error / (torch.abs(y) + 1e-8)
        exceeds = (abs_error > atol) & (rel_error > rtol)
        max_abs = max(
            max_abs, float(abs_error.max().item()) if abs_error.numel() else 0.0
        )
        max_rel = max(
            max_rel, float(rel_error.max().item()) if rel_error.numel() else 0.0
        )
        total += abs_error.numel()
        bad += int(exceeds.sum().item())

    matched_ratio = 1.0 - bad / total if total else 1.0
    required = 1.0 if required_matched_ratio is None else required_matched_ratio
    ok = matched_ratio >= required
    return ok, max_abs, max_rel, matched_ratio, "ok" if ok else "numerical mismatch"


def summarize(rows, group_axis: str):
    valid = [r for r in rows if r["passed"] and r["speedup"] and r["speedup"] > 0]
    print()
    print("Summary")
    print(f"  passed: {sum(r['passed'] for r in rows)}/{len(rows)}")
    # Failing workloads are reported explicitly and never counted as speedup 0;
    # they are simply excluded from the geomeans below.
    failed = [r for r in rows if not r["passed"]]
    if failed:
        kinds = {}
        for r in failed:
            kinds[r.get("status", "error")] = kinds.get(r.get("status", "error"), 0) + 1
        print("  failures: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        for r in failed:
            print(
                f"    FAIL [{r.get('status')}] {r['suite']}:{r['id']} {r.get('note')}"
            )
    if not valid:
        print("  no valid speedups")
        return

    def _geomean(vals):
        return math.exp(statistics.mean(math.log(s) for s in vals))

    speedups = [r["speedup"] for r in valid]
    print(f"  mean speedup: {statistics.mean(speedups):.4f}x")
    print(f"  min/max speedup: {min(speedups):.4f}x / {max(speedups):.4f}x")

    # Three headline geomeans split by size_class. The large geomean is the
    # primary ranking metric (production regime); the small geomean is a
    # reported guardrail for launch-overhead regressions; the all geomean is
    # kept for continuity. Refuse to summarize if any scored row is unclassified.
    unknown = [
        r for r in valid if r.get("size_class") not in size_policy.VALID_SIZE_CLASSES
    ]
    if unknown:
        raise size_policy.UnknownWorkload(
            f"{len(unknown)} scored workloads have no valid size_class; refusing to summarize"
        )
    large = [r["speedup"] for r in valid if r["size_class"] == size_policy.LARGE]
    small = [r["speedup"] for r in valid if r["size_class"] == size_policy.SMALL]
    print(f"  all geomean:   {_geomean(speedups):.4f}x (n={len(speedups)})")
    if large:
        print(f"  large geomean: {_geomean(large):.4f}x (n={len(large)})  [primary]")
    else:
        print("  large geomean: n/a (n=0)  [primary]")
    if small:
        print(f"  small geomean: {_geomean(small):.4f}x (n={len(small)})  [guardrail]")
    else:
        print("  small geomean: n/a (n=0)  [guardrail]")
    if group_axis:
        print(f"  by {group_axis}:")
        groups = {}
        for row in valid:
            groups.setdefault(row["axes"].get(group_axis), []).append(row)
        for value in sorted(groups):
            speedups_g = [r["speedup"] for r in groups[value]]
            print(
                f"    {value}: geo="
                f"{math.exp(statistics.mean(math.log(s) for s in speedups_g)):.4f}x "
                f"n={len(speedups_g)}"
            )


def run_benchmark(
    *,
    name: str,
    workloads: list,
    make_inputs,
    baseline_fn,
    candidate_fn,
    baseline_prepare_fn=None,
    candidate_prepare_fn=None,
    reference_fn=None,
    reference_prepare_fn=None,
    timing_baseline_fn=None,
    timing_baseline_prepare_fn=None,
    device: torch.device,
    warmup: int,
    iters: int,
    trials: int,
    atol: float,
    rtol: float,
    required_matched_ratio,
    group_axis: str,
):
    """Run one kernel over its workloads and report all/large/small geomeans.

    Variance and failure semantics:
      * Inputs are seeded deterministically per workload by each runner, so a
        repeated run reproduces the same shapes.
      * Each workload is timed over ``trials`` measurements (each with ``warmup``
        and ``iters`` internal iterations); the per-workload latency is the
        MEDIAN across trials, which is robust to occasional slow trials.
      * A workload that raises is recorded with status ``error``; one that fails
        the correctness gate is ``incorrect``; both are excluded from the
        geomeans and reported explicitly. Failures are never folded in as a
        speedup of 0.
    """
    print(f"OP: {name}")
    print(f"workloads: {len(workloads)}")
    print(f"device: {device}")
    print(f"environment: {capture_environment(device)}")
    print(f"warmup/iters/trials: {warmup}/{iters}/{trials}")
    print()

    rows = []
    timer_name = None
    for idx, entry in enumerate(workloads, start=1):
        axes = entry["axes"]
        suite = entry["suite"]
        wid = entry["id"]
        print(f"[{idx:03d}/{len(workloads):03d}] {suite}:{wid} axes={axes}")
        trial_baseline = []
        trial_kernel = []
        max_abs = 0.0
        max_rel = 0.0
        matched = 0.0
        passed = True
        note = "ok"

        for _ in range(trials):
            try:
                args = tuple(make_inputs(entry, device))
                candidate_input_args = clone_args(args)
                baseline_input_args = clone_args(args)
                candidate_args = (
                    candidate_input_args
                    if candidate_prepare_fn is None
                    else tuple(candidate_prepare_fn(*candidate_input_args))
                )
                baseline_args = (
                    baseline_input_args
                    if baseline_prepare_fn is None
                    else tuple(baseline_prepare_fn(*baseline_input_args))
                )
                reference_input_args = clone_args(args)
                reference_args = (
                    baseline_args
                    if reference_prepare_fn is None
                    else tuple(reference_prepare_fn(*reference_input_args))
                )
                correctness_fn = baseline_fn if reference_fn is None else reference_fn
                with torch.no_grad():
                    got = candidate_fn(*candidate_args)
                    ref = (
                        got
                        if correctness_fn is None
                        else correctness_fn(*reference_args)
                    )
                torch.cuda.synchronize(device)
                if correctness_fn is None:
                    ok, abs_err, rel_err, matched, note = (
                        True,
                        0.0,
                        0.0,
                        1.0,
                        "unchecked",
                    )
                else:
                    ok, abs_err, rel_err, matched, note = compare_outputs(
                        got, ref, atol, rtol, required_matched_ratio
                    )
            except Exception as exc:
                ok = False
                abs_err = float("inf")
                rel_err = float("inf")
                matched = 0.0
                note = f"{type(exc).__name__}: {exc}"
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if not ok:
                passed = False
                break
            try:
                timing_args = tuple(make_inputs(entry, device))
                timing_candidate_input_args = clone_args(timing_args)
                timing_baseline_input_args = clone_args(timing_args)
                timing_candidate_args = (
                    timing_candidate_input_args
                    if candidate_prepare_fn is None
                    else tuple(candidate_prepare_fn(*timing_candidate_input_args))
                )
                timing_baseline_args = (
                    timing_baseline_input_args
                    if baseline_prepare_fn is None
                    else tuple(baseline_prepare_fn(*timing_baseline_input_args))
                )
                timing_reference_args = timing_baseline_args
                if reference_prepare_fn is not None:
                    timing_reference_input_args = clone_args(timing_args)
                    timing_reference_args = tuple(
                        reference_prepare_fn(*timing_reference_input_args)
                    )
                timing_base_fn = (
                    baseline_fn if timing_baseline_fn is None else timing_baseline_fn
                )
                if timing_baseline_prepare_fn is not None:
                    timing_baseline_args = tuple(
                        timing_baseline_prepare_fn(*timing_baseline_input_args)
                    )
                if timing_base_fn is None:
                    base_ms, timer = None, "candidate_only"
                else:
                    base_ms, timer = time_fn(
                        timing_base_fn, timing_baseline_args, device, warmup, iters
                    )
                kernel_ms, timer = time_fn(
                    candidate_fn, timing_candidate_args, device, warmup, iters
                )
                if correctness_fn is not None:
                    with torch.no_grad():
                        hot_ref = correctness_fn(*timing_reference_args)
                        hot_got = candidate_fn(*timing_candidate_args)
                    torch.cuda.synchronize(device)
                    hot_ok, hot_abs, hot_rel, hot_matched, hot_note = compare_outputs(
                        hot_got, hot_ref, atol, rtol, required_matched_ratio
                    )
                    max_abs = max(max_abs, hot_abs)
                    max_rel = max(max_rel, hot_rel)
                    matched = min(matched, hot_matched)
                    if not hot_ok:
                        passed = False
                        note = hot_note
                        break
            except Exception as exc:
                passed = False
                note = f"{type(exc).__name__}: {exc}"
                break
            timer_name = timer_name or timer
            if base_ms is not None:
                trial_baseline.append(base_ms)
            trial_kernel.append(kernel_ms)

        if passed and trial_baseline:
            # Median across trials is robust to occasional slow trials (clocks,
            # contention) versus the mean.
            baseline_ms = statistics.median(trial_baseline)
            kernel_ms = statistics.median(trial_kernel)
            speedup = baseline_ms / kernel_ms
            print(f"  baseline: {baseline_ms:.6f} ms")
            print(f"  kernel:   {kernel_ms:.6f} ms")
            print(f"  speedup:  {speedup:.4f}x")
            print(
                f"  correct:  PASS max_abs={max_abs:.3e} "
                f"max_rel={max_rel:.3e} matched={matched:.4f}"
            )
        elif passed and trial_kernel:
            baseline_ms = None
            kernel_ms = statistics.median(trial_kernel)
            speedup = None
            print("  baseline: unavailable")
            print(f"  kernel:   {kernel_ms:.6f} ms")
            print(
                f"  correct:  PASS max_abs={max_abs:.3e} "
                f"max_rel={max_rel:.3e} matched={matched:.4f}"
            )
        else:
            baseline_ms = None
            kernel_ms = None
            speedup = None
            print(
                f"  correct:  FAIL {note} max_abs={max_abs:.3e} "
                f"max_rel={max_rel:.3e} matched={matched:.4f}"
            )

        if not passed:
            # Correctness-gate failures (including NaN/Inf outputs, rejected by
            # compare_outputs) are "incorrect"; anything else is a runtime error.
            status = "incorrect" if note in _CORRECTNESS_FAIL_NOTES else "error"
        elif speedup is None:
            status = "candidate_only"
        else:
            status = "ok"
        rows.append(
            {
                "suite": suite,
                "id": wid,
                "axes": axes,
                "size_class": size_policy.classify(name, axes),
                "passed": passed,
                "status": status,
                "note": note,
                "baseline_ms": baseline_ms,
                "kernel_ms": kernel_ms,
                "speedup": speedup,
            }
        )
        print()

    if timer_name:
        print(f"timer: {timer_name}")
    summarize(rows, group_axis)
    _emit_output_jsonl(name, rows)
    return rows


def _emit_output_jsonl(name: str, rows: list) -> None:
    """Append this run's per-workload rows to ``BENCH_OUTPUT_JSONL`` when set.

    The out-of-docker judge runs the standalone bench with ``BENCH_OUTPUT_JSONL``
    pointing at a results file it then reads back and scores. One JSONL line per
    ``run_benchmark`` call: ``{"op","workload_count","rows":[...]}``. ``workload_count``
    lets the judge fail closed on a truncated/partial run (see gpu_runner.parse_verified).
    """
    path = os.environ.get("BENCH_OUTPUT_JSONL")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(
            json.dumps({"op": name, "workload_count": len(rows), "rows": rows}) + "\n"
        )
