"""Pinned diffusion references for the H200 optimization campaign."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import torch
from _pilot_adapter import Adapter, bind


class DiffusionAdapter(Adapter):
    def __init__(self, definition):
        self.definition = definition
        super().__init__(
            definition["targets"],
            {op: ("exact", "pinned current kernel") for op in definition["targets"]},
        )
        self._verified = False

    def target(self, op):
        if not self._verified:
            for module_name, expected in self.definition["source_sha256"].items():
                module = importlib.import_module(module_name)
                source = Path(module.__file__)
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"pinned baseline source changed: {source}")
            self._verified = True
        return super().target(op)

    def make_inputs(self, row, *, device, seed):
        op, kwargs = super().make_inputs(row, device=device, seed=seed)
        # Only source-declared index domains are repaired; floating values stay
        # random, and are regenerated independently for baseline and candidate.
        for name, spec in row["kwargs"].items():
            if isinstance(spec, dict):
                if "values" in spec:
                    value = kwargs[name]
                    value.copy_(
                        torch.tensor(
                            spec["values"], dtype=value.dtype, device=value.device
                        ).reshape(value.shape)
                    )
                if "bounded_int" in spec:
                    kwargs[name].remainder_(int(spec["bounded_int"]))
        return op, kwargs

    def output_tensors(self, inputs):
        op, kwargs = inputs
        return tuple(
            kwargs[name]
            for name in self.definition.get("output_aliases", {}).get(op, [])
        )

    def poison_outputs(self, inputs):
        # These aliases are read/write operands, not output-only buffers.
        # Poisoning x before indexed modulation would corrupt the input. The
        # harness still checks identity and perturbs live inputs independently.
        pass

    def compare_input_state(self, actual, expected):
        actual_op, actual_kwargs = actual
        expected_op, expected_kwargs = expected
        if actual_op != expected_op or actual_kwargs.keys() != expected_kwargs.keys():
            return False, "candidate changed its input mapping"
        for name, ref in expected_kwargs.items():
            if not isinstance(ref, torch.Tensor):
                continue
            got = actual_kwargs[name]
            if not isinstance(got, torch.Tensor):
                return False, f"input {name}: no longer a tensor"
            if (
                got.shape != ref.shape
                or got.dtype != ref.dtype
                or got.stride() != ref.stride()
            ):
                return False, f"input {name}: metadata changed"
            if not torch.equal(
                got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
            ):
                return False, f"input {name}: unexpected mutation"
        return True, "input mutation matches baseline"

    def check_output_aliases(self, result, inputs):
        op, kwargs = inputs
        allowed = set(self.definition.get("output_aliases", {}).get(op, []))
        outputs = tuple(result) if isinstance(result, (tuple, list)) else (result,)
        for output in outputs:
            if not isinstance(output, torch.Tensor):
                continue
            for name, value in kwargs.items():
                if (
                    isinstance(value, torch.Tensor)
                    and name not in allowed
                    and output.untyped_storage().data_ptr()
                    == value.untyped_storage().data_ptr()
                ):
                    return False, f"output unexpectedly aliases input {name}"
        return True, "output storage aliases match contract"

    def compare(self, actual, expected, row):
        actuals = tuple(actual) if isinstance(actual, (tuple, list)) else (actual,)
        expecteds = (
            tuple(expected) if isinstance(expected, (tuple, list)) else (expected,)
        )
        if len(actuals) != len(expecteds):
            return False, "wrong number of outputs"
        for index, (got, ref) in enumerate(zip(actuals, expecteds, strict=True)):
            if not isinstance(ref, torch.Tensor) or not isinstance(got, torch.Tensor):
                return False, f"output {index}: task requires tensor, not fallback"
            if (
                got.shape != ref.shape
                or got.dtype != ref.dtype
                or got.device != ref.device
            ):
                return False, f"output {index}: shape/dtype/device mismatch"
            if got.stride() != ref.stride():
                return False, f"output {index}: layout mismatch"
            if not torch.equal(
                got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
            ):
                return False, f"output {index}: not equal to pinned kernel"
        return True, "exact values, dtype and layout"

    def axes(self, row):
        elements = max(
            math.prod(spec["shape"])
            for spec in row["kwargs"].values()
            if isinstance(spec, dict) and "shape" in spec
        )
        return {"elements": elements}

    def size_class(self, row):
        return "large" if self.axes(row)["elements"] >= 1048576 else "small"


def bind_diffusion(namespace, name):
    path = Path(__file__).resolve().parents[1] / "definitions" / f"{name}.json"
    adapter = DiffusionAdapter(json.loads(path.read_text()))
    bind(namespace, adapter)
    namespace["poison_outputs"] = adapter.poison_outputs
    namespace["compare_input_state"] = adapter.compare_input_state
    namespace["check_output_aliases"] = adapter.check_output_aliases
