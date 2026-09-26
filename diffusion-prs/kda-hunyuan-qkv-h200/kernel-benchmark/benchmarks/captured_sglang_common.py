"""Shared helpers for standalone tasks imported from KDA-Pilot captures."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch

_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    # NVFP4 activations and weights travel as two e2m1 nibbles per byte, so the
    # captured spec is uint8 with the K dimension already halved.
    "torch.uint8": torch.uint8,
    # The B300 packages add three: `has_initial_state` on the Mamba-2 prefill is a
    # per-sequence bool flag, and the diffusion RoPE tables travel as complex64.
    "torch.bool": torch.bool,
    "torch.complex64": torch.complex64,
    "torch.float16": torch.float16,
}
_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def dtype_from_spec(spec: dict) -> torch.dtype:
    try:
        return _DTYPES[spec["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported captured dtype: {spec.get('dtype')!r}") from exc


def empty_from_spec(spec: dict, device: torch.device) -> torch.Tensor:
    shape = tuple(int(value) for value in spec["shape"])
    stride = tuple(int(value) for value in spec.get("stride", ()))
    storage_offset = int(spec.get("storage_offset", 0))
    dtype = dtype_from_spec(spec)
    if stride:
        storage_size = storage_offset + 1
        if shape and all(size > 0 for size in shape):
            storage_size = (
                storage_offset
                + 1
                + sum(
                    (size - 1) * step for size, step in zip(shape, stride, strict=True)
                )
            )
        base = torch.empty(storage_size, dtype=dtype, device=device)
        return base.as_strided(shape, stride, storage_offset)
    return torch.empty(shape, dtype=dtype, device=device)


def _random_contiguous(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    *,
    positive: bool,
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        values = torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).clamp_(-2.0, 2.0)
        if positive:
            values = values.abs_().mul_(0.05).add_(0.01)
        return values.to(dtype)
    if dtype == torch.bool:
        # A flag array. Drawn all-True rather than at random: every one of these is a
        # "does this sequence carry state" predicate, and the recorded call had state
        # for the sequences it was given. A random draw would silently benchmark the
        # cheaper half of the kernel on half the rows.
        return torch.ones(shape, dtype=torch.bool, device=device)
    if dtype == torch.complex64:
        real = torch.randn(
            shape, dtype=torch.float32, device=device, generator=generator
        )
        imag = torch.randn(
            shape, dtype=torch.float32, device=device, generator=generator
        )
        if positive:
            real, imag = real.abs_(), imag.abs_()
        return torch.complex(real, imag)
    if dtype in (torch.float32, torch.bfloat16, torch.float16):
        if positive:
            return (
                torch.rand(
                    shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                * 0.09
                + 0.01
            ).to(dtype)
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(
            0,
            1024,
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    if dtype == torch.uint8:
        # Packed FP4 pairs and swizzled e4m3 scale blocks are opaque bytes to the
        # kernel: any byte pattern is a legal encoding, so draw the full range.
        return torch.randint(
            0,
            256,
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    raise ValueError(f"unsupported captured dtype: {dtype}")


def random_from_spec(
    spec: dict,
    device: torch.device,
    generator: torch.Generator,
    *,
    positive: bool = False,
    cache_key: str | None = None,
) -> torch.Tensor:
    shape = tuple(int(value) for value in spec["shape"])
    dtype = dtype_from_spec(spec)
    stride = tuple(int(value) for value in spec.get("stride", ()))
    storage_offset = int(spec.get("storage_offset", 0))
    key = (cache_key, str(device), dtype, shape, stride, storage_offset, positive)
    if cache_key is not None and key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]

    values = _random_contiguous(
        shape,
        dtype,
        device,
        generator,
        positive=positive,
    )
    if stride and (values.stride() != stride or storage_offset != 0):
        result = empty_from_spec(spec, device)
        result.copy_(values)
    else:
        result = values
    if cache_key is not None:
        _TENSOR_CACHE[key] = result
    return result


def clone_result(result):
    if isinstance(result, torch.Tensor):
        clone = empty_from_spec(
            {
                "shape": result.shape,
                "stride": result.stride(),
                "storage_offset": result.storage_offset(),
                "dtype": str(result.dtype),
            },
            result.device,
        )
        clone.copy_(result)
        return clone
    if isinstance(result, tuple):
        return tuple(clone_result(value) for value in result)
    if isinstance(result, list):
        return [clone_result(value) for value in result]
    return result


def poison_result(result) -> None:
    values: Iterable = result if isinstance(result, (tuple, list)) else (result,)
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            value.fill_(float("nan"))
        elif value.dtype == torch.uint8:
            value.fill_(0xA5)
        else:
            value.fill_(-17)


def _mutation_views(tensor: torch.Tensor) -> tuple:
    """Two interleaved in-place handles on `tensor`, whatever its layout.

    A contiguous tensor flattens with `view(-1)`. A strided one -- a transposed
    view, which is what several captured rows legitimately pass -- cannot:
    `view(-1)` raises "at least one dimension spans across two contiguous
    subspaces", and `reshape(-1)` silently returns a *copy*, so mutating it would
    leave the real argument untouched and disarm this guard without saying so.
    Index along the last dimension instead, which stays a view for any layout.
    """
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        return flat, flat[::2], flat[1::2]
    if tensor.shape[-1] >= 2:
        return tensor, tensor[..., ::2], tensor[..., 1::2]
    return tensor, tensor, None


def mutate_tensor(tensor: torch.Tensor) -> None:
    """Perturb a tensor in place so a cached result cannot pass as a live one.

    Torch has no elementwise ``mul_`` for ``float8_e4m3fn`` (it raises
    ``NotImplementedError: "mul_cuda" not implemented``), and the FP8 projection
    rows hand the activation in exactly that dtype, so the scaling round-trips
    through fp32. Packed-byte tensors carry no arithmetic meaning at all -- any
    byte is a legal FP4 pair -- so those are perturbed bitwise instead.
    """
    if tensor.numel() == 0:
        return
    whole, even, odd = _mutation_views(tensor)
    if tensor.dtype == torch.uint8:
        even.bitwise_xor_(0x55)
        return
    if tensor.dtype == torch.float8_e4m3fn:
        # `.float()` is a contiguous copy of the same shape, so its own halves are
        # flat; the result is written back through the original layout.
        values = tensor.float()
        _, value_even, value_odd = _mutation_views(values)
        value_even.mul_(-0.5)
        if value_odd is not None:
            value_odd.mul_(1.5)
        tensor.copy_(values.to(tensor.dtype))
        return
    even.mul_(-0.5)
    if odd is not None:
        odd.mul_(1.5)


def _flatten(result) -> list:
    if isinstance(result, (tuple, list)):
        return list(result)
    return [result]


def compare_results(
    actual,
    expected,
    *,
    atol: float,
    rtol: float,
    exact_float8: bool = False,
) -> tuple[bool, str]:
    actual_values = _flatten(actual)
    expected_values = _flatten(expected)
    if len(actual_values) != len(expected_values):
        return False, "wrong number of outputs"

    for index, (got, ref) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if not isinstance(got, torch.Tensor) or not isinstance(ref, torch.Tensor):
            if got != ref:
                return False, f"non-tensor output {index} differs"
            continue
        if got.shape != ref.shape:
            return False, f"output {index} shape differs: {got.shape} != {ref.shape}"
        if got.dtype != ref.dtype:
            return False, f"output {index} dtype differs: {got.dtype} != {ref.dtype}"
        if got.stride() != ref.stride():
            return (
                False,
                f"output {index} stride differs: {got.stride()} != {ref.stride()}",
            )
        if got.storage_offset() != ref.storage_offset():
            note = (
                f"output {index} storage offset differs: "
                f"{got.storage_offset()} != {ref.storage_offset()}"
            )
            return False, note
        if got.dtype in (torch.int32, torch.int64, torch.bool, torch.uint8) or (
            exact_float8 and got.dtype == torch.float8_e4m3fn
        ):
            lhs = got.view(torch.uint8) if got.dtype == torch.float8_e4m3fn else got
            rhs = ref.view(torch.uint8) if ref.dtype == torch.float8_e4m3fn else ref
            if not torch.equal(lhs, rhs):
                return False, f"output {index} is not bit-exact"
            continue
        got_float = got.float()
        ref_float = ref.float()
        if not bool(torch.isfinite(got_float).all().item()):
            return False, f"output {index} contains NaN or Inf"
        if not torch.allclose(got_float, ref_float, atol=atol, rtol=rtol):
            max_abs = float((got_float - ref_float).abs().max().item())
            return False, f"output {index} numerical mismatch max_abs={max_abs:.3e}"
    return True, "ok"


def transposed_spec(spec: dict) -> dict:
    """The same tensor spec with its last two dimensions swapped.

    Several captured weights are recorded the way the checkpoint stores them,
    `[in_features, out_features]`, while the kernel that consumes them wants the
    opposite. Swapping the *spec* rather than transposing the materialized tensor
    matters: these weights reach 248320x2560, so `.t().contiguous()` would
    allocate a second copy on every `make_inputs`, while a swapped spec is drawn
    once and cached like any other reused argument.
    """
    shape = [int(value) for value in spec["shape"]]
    if len(shape) < 2:
        return dict(spec)
    shape[-1], shape[-2] = shape[-2], shape[-1]
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return {**spec, "shape": shape, "stride": stride, "is_contiguous": True}


def kwargs_from_row(
    row: dict,
    *,
    device: torch.device,
    seed: int,
    positive: tuple[str, ...] = (),
    reuse: tuple[str, ...] = (),
    transpose: tuple[str, ...] = (),
) -> dict:
    """Materialize one named-argument captured row into call kwargs.

    Rows imported from a KDA-Pilot ``nvidia_agent_tasks`` package key every
    argument by parameter name: a tensor spec becomes a tensor drawn from the
    recorded shape/dtype/stride, and a scalar (including ``None``) is passed
    straight through.

    ``transpose`` names arguments whose recorded orientation is not the one the
    kernel takes; see :func:`transposed_spec`.

    ``positive`` names arguments that must not straddle zero -- scales and
    global-scale factors, where a negative draw would make the reference itself
    meaningless. ``reuse`` names arguments cached across calls: weights are the
    largest tensors in these rows and are identical on every invocation, so
    redrawing them per ``make_inputs`` would dominate setup and, for the FP4 GEMM
    rows, allocate hundreds of megabytes repeatedly.

    Each argument gets its **own** generator, seeded from the row seed and the
    argument name. One shared generator would couple every draw to iteration
    order *and* to cache state: a ``reuse`` argument served from the cache does
    not consume generator state, so the second ``make_inputs`` for the same row
    would advance differently and hand every later argument different values.
    The harness builds inputs twice per row and compares the baseline against
    itself, so that desync surfaces as the baseline failing its own correctness
    gate -- observed as ``plain run: output 0 numerical mismatch`` on three of
    four FP8 rows before this was per-argument.
    """
    kwargs = {}
    for name, value in row["kwargs"].items():
        if not isinstance(value, dict):
            kwargs[name] = value
            continue
        if "shape" not in value or "dtype" not in value:
            # Not every dict is a tensor spec. The MoE rows pass the Triton launch
            # `config` as a plain dict, and an argument the capture could not serialize
            # is recorded as {"repr": "<ClassName>"} for the task's repair hook to
            # replace. Drawing a tensor from either is a KeyError on 'shape', which is
            # what every lfm25 and glm45 row failed with.
            kwargs[name] = value
            continue
        if name in transpose:
            value = transposed_spec(value)
        material = f"{row['id']}\0{name}".encode()
        argument_seed = (
            int.from_bytes(hashlib.sha256(material).digest()[:8], "little") ^ seed
        )
        generator = torch.Generator(device=device).manual_seed(
            argument_seed % (2**63 - 1)
        )
        kwargs[name] = random_from_spec(
            value,
            device,
            generator,
            positive=name in positive,
            cache_key=f"{row['function']}:{name}" if name in reuse else None,
        )
    return kwargs
