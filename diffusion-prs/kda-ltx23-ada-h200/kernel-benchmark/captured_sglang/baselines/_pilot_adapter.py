"""One adapter for every task imported from a KDA-Pilot `nvidia_agent_tasks` package.

The imported packages differ only in three things: which entry point each op
resolves to, which arguments a random draw cannot stand in for, and what the gate
is. Everything else - materializing a row, picking the size axes, dispatching on
the row's `op` - is identical, so it lives here once and each task file is a
declaration rather than a copy.

A task file supplies:

    TARGETS      op -> "module.attr" (or "module.Class.method"), resolved lazily so
                 an import error names the op it belongs to
    TOLERANCES   op -> (rtol, atol, "source") or ("exact", "source")
    REPAIR       op -> callable(kwargs) applied once, at build time (never inside a
                 timed region: input repair measured as kernel time cost an identity
                 candidate 1.75x upstream before this moved out)
    REUSE        argument names drawn once and cached - weights, whose redraw would
                 dominate setup
    POSITIVE     argument names that must not straddle zero - scales, where a
                 negative draw makes the reference itself meaningless

and nothing else.
"""

from __future__ import annotations

import importlib

import torch
from captured_sglang_common import compare_results, kwargs_from_row


def resolve(path: str):
    """`module.attr` or `module.Class.method` -> the callable, imported on demand."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        target = module
        for attr in parts[cut:]:
            target = getattr(target, attr)
        return target
    raise ImportError(f"cannot resolve {path!r}")


class Adapter:
    def __init__(self, targets, tolerances, repair=None, reuse=(), positive=()):
        self.targets = targets
        self.tolerances = tolerances
        self.repair = repair or {}
        self.reuse = tuple(reuse)
        self.positive = tuple(positive)
        self._cache: dict = {}

    # -- dispatch ---------------------------------------------------------- #
    def target(self, op: str):
        if op not in self._cache:
            if op not in self.targets:
                raise ValueError(f"unknown captured op: {op!r}")
            self._cache[op] = resolve(self.targets[op])
        return self._cache[op]

    def run(self, op: str, kwargs: dict):
        return self.target(op)(**kwargs)

    def prepare_benchmark(self, op: str, kwargs: dict):
        return self.target(op), (), kwargs

    # -- inputs ------------------------------------------------------------ #
    def make_inputs(self, row: dict, *, device, seed: int):
        kwargs = kwargs_from_row(
            row, device=device, seed=seed, positive=self.positive, reuse=self.reuse
        )
        fix = self.repair.get(row["op"])
        if fix is not None:
            kwargs = fix(kwargs)
        return row["op"], kwargs

    def output_tensors(self, _inputs: tuple):
        # These ops allocate their own outputs; the ones that write through an
        # argument declare it in their task file by overriding this.
        return ()

    def mutate_inputs(self, inputs: tuple) -> None:
        """Perturb the first floating-point activation, so a cached result cannot pass.

        Weights are skipped: they are `REUSE`-cached and shared with later rows, and
        perturbing them would change what every subsequent row measures.
        """
        from captured_sglang_common import mutate_tensor

        _, kwargs = inputs
        for name, value in kwargs.items():
            if name in self.reuse or not torch.is_tensor(value):
                continue
            if value.is_floating_point() and value.numel() > 1:
                mutate_tensor(value)
                return

    # -- gate -------------------------------------------------------------- #
    def compare(self, actual, expected, row: dict):
        spec = self.tolerances.get(row["op"])
        if spec is None:
            raise ValueError(f"no tolerance declared for op {row['op']!r}")
        if spec[0] == "exact":
            return compare_results(
                actual, expected, atol=0.0, rtol=0.0, exact_float8=True
            )
        rtol, atol = spec[0], spec[1]
        return compare_results(actual, expected, atol=atol, rtol=rtol)

    # -- size -------------------------------------------------------------- #
    def axes(self, row: dict) -> dict:
        """Token count and the widest recorded dimension.

        Every one of these kernels is sized by how many tokens it is handed and how
        wide the row it walks is, so those two are enough to bucket a row - and both
        come off the recorded specs rather than a per-task convention.
        """
        tokens, widest = 0, 0
        for spec in row["kwargs"].values():
            if not (isinstance(spec, dict) and spec.get("shape")):
                continue
            shape = [int(v) for v in spec["shape"]]
            if not shape:
                continue
            tokens = max(tokens, shape[-2] if len(shape) >= 3 else shape[0])
            widest = max(widest, max(shape))
        return {"tokens": int(tokens), "widest": int(widest)}

    def size_class(self, row: dict) -> str:
        return "large" if self.axes(row)["tokens"] >= 8 else "small"


def bind(module_globals: dict, adapter: Adapter) -> None:
    """Export the harness's expected module-level symbols from one adapter."""
    module_globals.update(
        run=adapter.run,
        prepare_benchmark=adapter.prepare_benchmark,
        make_inputs=adapter.make_inputs,
        output_tensors=adapter.output_tensors,
        mutate_inputs=adapter.mutate_inputs,
        compare=adapter.compare,
        axes=adapter.axes,
        size_class=adapter.size_class,
    )
