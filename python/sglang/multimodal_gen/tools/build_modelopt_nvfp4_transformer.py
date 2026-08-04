"""Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer.

This tool keeps the ModelOpt-exported NVFP4 tensors for most transformer
modules, but can replace a validated subset of numerically sensitive modules
with their original BF16 tensors from the base transformer checkpoint.

It supports both FLUX.1-style ModelOpt HF exports and compressed native
MiniMax H3 ``mto.save`` checkpoints.  In both paths:
- the base pipeline should remain separate from the quantized transformer
- fallback BF16 modules are model-family specific
- the serialized FP4 weight byte order may already match the runtime kernel

MiniMax H3 example::

    python -m sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer \
        --base-transformer-dir /path/to/MiniMax-H3/FL2VA/transformer \
        --modelopt-backbone-ckpt /tmp/minimax-h3-nvfp4/backbone.pt \
        --pattern-preset minimax-h3-nvfp4 \
        --output-dir /tmp/minimax-h3-nvfp4/transformer

On a memory-constrained Blackwell machine, capture input maxima and exact weight
double scales during a normal BF16 generation, then quantize the original
checkpoint one shard at a time::

    python -m sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer \
        --base-transformer-dir /path/to/MiniMax-H3/FL2VA/transformer \
        --modelopt-backbone-ckpt /tmp/h3-amax/calibration-state.pt \
        --allow-unquantized-source --offline-quant-device cuda \
        --pattern-preset minimax-h3-nvfp4 \
        --output-dir /tmp/minimax-h3-nvfp4/transformer
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]

DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS = [
    "transformer_blocks.*.norm1.linear*",
    "transformer_blocks.*.norm1_context.linear*",
    "transformer_blocks.*.ff.net.0.proj*",
    "transformer_blocks.*.ff.net.2*",
    "transformer_blocks.*.ff_context.net.0.proj*",
    "transformer_blocks.*.ff_context.net.2*",
    "single_transformer_blocks.*.norm.linear*",
    "single_transformer_blocks.*.proj_mlp*",
]
DEFAULT_MINIMAX_H3_NVFP4_FALLBACK_PATTERNS = [
    "video_patch_proj",
    "audio_patch_proj",
    "time_embedder.*",
    "final_layer.video_out",
    "final_layer.audio_out",
    "condition_proj",
    "token_refiner.*",
    "blocks.0.attn.*",
    "blocks.*.adaln_proj.*",
    "blocks.49.attn.*",
    "final_layer.adaln_proj.*",
]

_TENSOR_MODULE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
    ".bias",
)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    for relative in ("FL2VA/transformer", "transformer"):
        transformer_dir = candidate / relative
        if (transformer_dir / "config.json").is_file():
            return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _resolve_backbone_ckpt(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return str(candidate)
    backbone_path = candidate / "backbone.pt"
    if backbone_path.is_file():
        return str(backbone_path)
    raise FileNotFoundError(f"Could not resolve backbone.pt from: {path}")


def _find_index_file(model_dir: str) -> str | None:
    for filename in INDEX_FILENAMES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return filename

    matches = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors.index.json")
    )
    return matches[0] if matches else None


def _load_weight_map(model_dir: str) -> tuple[dict[str, str], str | None]:
    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    index_filename = f"{Path(shard_name).stem}.safetensors.index.json"
    return weight_map, index_filename


def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _write_config(model_dir: Path, config: Mapping[str, object]) -> None:
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_non_shard_files(source_dir: str, output_dir: str) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in ignored:
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _load_selected_tensors(
    model_dir: str,
    weight_map: Mapping[str, str],
    tensor_names: Iterable[str],
):
    tensors = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        names_by_file[weight_map[name]].append(name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def _module_name_for_tensor(tensor_name: str) -> str:
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _matches_any_pattern(module_name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if re.fullmatch(regex_str, module_name):
            return True
    return False


def _preset_patterns(pattern_preset: str) -> list[str]:
    if pattern_preset == "none":
        return []
    if pattern_preset == "flux1-nvfp4":
        return list(DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS)
    if pattern_preset == "minimax-h3-nvfp4":
        return list(DEFAULT_MINIMAX_H3_NVFP4_FALLBACK_PATTERNS)
    raise ValueError(f"Unsupported pattern preset: {pattern_preset}")


def _updated_quant_config(
    source_config: Mapping[str, object],
    *,
    fallback_patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    output_config = json.loads(json.dumps(source_config))
    quant_config = output_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("Expected a flat quantization_config dict in config.json.")
    if (
        quant_config.get("quant_method") != "modelopt"
        or "FP4" not in str(quant_config.get("quant_algo", "")).upper()
    ):
        raise ValueError(
            "This tool only supports ModelOpt diffusion NVFP4 exports "
            "(quant_method=modelopt, quant_algo=FP4/NVFP4)."
        )

    ignore_patterns = list(quant_config.get("ignore", []) or [])
    for pattern in fallback_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)

    quant_config["ignore"] = ignore_patterns
    quant_config.setdefault(
        "quant_type", str(quant_config.get("quant_algo", "")).upper()
    )
    quant_config["swap_weight_nibbles"] = swap_weight_nibbles
    return output_config


def _load_modelopt_backbone_state(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        _resolve_backbone_ckpt(path), map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("ModelOpt backbone checkpoint has no model_state_dict")
    return state


def _modelopt_nvfp4_modules(
    state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    modules: dict[str, dict[str, torch.Tensor]] = {}
    scale_suffix = ".weight_quantizer._scale"
    for name, block_scale in state.items():
        if not name.endswith(scale_suffix):
            continue
        module_name = name[: -len(scale_suffix)]
        keys = {
            "weight": f"{module_name}.weight",
            "weight_scale_2": f"{module_name}.weight_quantizer._double_scale",
            "input_amax": f"{module_name}.input_quantizer._amax",
        }
        missing = [key for key in keys.values() if key not in state]
        if missing:
            raise ValueError(
                f"Incomplete compressed ModelOpt NVFP4 state for {module_name}: "
                f"missing {missing}"
            )
        packed_weight = state[keys["weight"]]
        if packed_weight.dtype != torch.uint8:
            raise ValueError(
                f"Expected packed uint8 NVFP4 weight for {module_name}, "
                f"got {packed_weight.dtype}"
            )
        modules[module_name] = {
            "weight": packed_weight.contiguous(),
            "weight_scale": block_scale.contiguous(),
            "weight_scale_2": state[keys["weight_scale_2"]]
            .to(torch.float32)
            .contiguous(),
            # ModelOpt NVFP4 uses an E2M1 maximum of 6 and normalizes the
            # activation scale into E4M3's 448 range.
            "input_scale": (
                state[keys["input_amax"]].to(torch.float32) / (6.0 * 448.0)
            ).contiguous(),
        }
    if not modules:
        raise ValueError(
            "No compressed NVFP4 modules found in ModelOpt backbone checkpoint"
        )
    return modules


def _modelopt_input_amax_map(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    suffix = ".input_quantizer._amax"
    return {
        name[: -len(suffix)]: value.detach().to(torch.float32).reshape(()).cpu()
        for name, value in state.items()
        if name.endswith(suffix)
    }


def _modelopt_weight_double_scale_map(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    suffix = ".weight_quantizer._double_scale"
    return {
        name[: -len(suffix)]: value.detach().to(torch.float32).reshape(()).cpu()
        for name, value in state.items()
        if name.endswith(suffix)
    }


def _quantize_nvfp4_weight(
    weight: torch.Tensor,
    *,
    chunk_rows: int = 1024,
    double_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ModelOpt's dynamic NVFP4 weight compression path."""
    if weight.ndim < 2 or weight.shape[-1] % 16:
        raise ValueError(
            "NVFP4 weights must have at least two dimensions and an input "
            f"dimension divisible by 16, got {tuple(weight.shape)}"
        )
    if chunk_rows <= 0:
        raise ValueError(f"NVFP4 chunk_rows must be positive, got {chunk_rows}")

    weight = weight.contiguous()
    input_dim = weight.shape[-1]
    matrix = weight.reshape(-1, input_dim)
    packed_weight = torch.empty(
        (matrix.shape[0], input_dim // 2), dtype=torch.uint8, device=weight.device
    )
    all_block_scales = torch.empty(
        (matrix.shape[0], input_dim // 16),
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )

    if double_scale is None:
        global_amax = torch.zeros((), dtype=torch.float32, device=weight.device)
        for start in range(0, matrix.shape[0], chunk_rows):
            current = matrix[start : start + chunk_rows].detach().abs().amax()
            global_amax = torch.maximum(global_amax, current.to(torch.float32))
        double_scale = global_amax / (_E2M1_MAX * _E4M3_MAX)
    else:
        double_scale = (
            double_scale.detach()
            .to(device=weight.device, dtype=torch.float32)
            .reshape(())
        )
    if not torch.isfinite(double_scale) or double_scale <= 0:
        raise ValueError("NVFP4 weight must have a finite, positive absolute maximum")

    bounds = torch.tensor(_E2M1_BOUNDS, dtype=torch.float32, device=weight.device)
    odd_bounds = bounds[[1, 3, 5]]
    for start in range(0, matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, matrix.shape[0])
        blocked = matrix[start:stop].reshape(stop - start, -1, 16)
        block_amax = blocked.detach().abs().amax(dim=-1).to(torch.float32)
        block_scale = block_amax / (_E2M1_MAX * double_scale)
        block_scale = torch.where(
            block_scale == 0, torch.ones_like(block_scale), block_scale
        )
        block_scale = block_scale.clamp(min=2**-9, max=_E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        scaled = blocked / (block_scale.to(torch.float32) * double_scale).unsqueeze(-1)
        scaled = scaled.reshape(stop - start, input_dim)
        absolute = scaled.abs()
        ordinal = torch.searchsorted(bounds, absolute, out_int32=True).to(torch.uint8)
        tie_adjustment = torch.any(absolute.unsqueeze(-1) == odd_bounds, dim=-1).to(
            torch.uint8
        )
        q_weight = ((scaled < 0).to(torch.uint8) << 3) + ordinal + tie_adjustment
        packed_weight[start:stop] = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
        all_block_scales[start:stop] = block_scale

    return (
        packed_weight.reshape(*weight.shape[:-1], input_dim // 2).contiguous(),
        all_block_scales.reshape(*weight.shape[:-1], input_dim // 16).contiguous(),
        double_scale.to(torch.float32).contiguous(),
    )


def _h3_qkv_runtime_rows_to_checkpoint_rows(
    tensor: torch.Tensor, *, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Undo H3's load-time [head,qkv] -> [q_all,k_all,v_all] permutation."""
    inner_dim = num_heads * head_dim
    if tensor.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"MiniMax H3 qkv tensor has {tensor.shape[0]} rows; "
            f"expected {3 * inner_dim}"
        )
    rest = tensor.shape[1:]
    q, k, v = tensor.split(inner_dim, dim=0)
    return (
        torch.stack(
            (
                q.reshape(num_heads, head_dim, *rest),
                k.reshape(num_heads, head_dim, *rest),
                v.reshape(num_heads, head_dim, *rest),
            ),
            dim=1,
        )
        .reshape(3 * inner_dim, *rest)
        .contiguous()
    )


def _maybe_restore_h3_qkv_checkpoint_layout(
    module_name: str,
    tensors: Mapping[str, torch.Tensor],
    source_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    if source_config.get(
        "_class_name"
    ) != "MiniMaxH3DiTModel" or not module_name.endswith(".attn.qkv_proj"):
        return dict(tensors)
    num_heads = int(source_config["num_attention_heads"])
    head_dim = int(source_config["attention_head_dim"])
    restored = dict(tensors)
    for key in ("weight", "weight_scale"):
        restored[key] = _h3_qkv_runtime_rows_to_checkpoint_rows(
            restored[key], num_heads=num_heads, head_dim=head_dim
        )
    return restored


def _direct_nvfp4_output_config(
    source_config: Mapping[str, object],
    *,
    fallback_patterns: Sequence[str],
    auto_ignored_modules: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    output_config = json.loads(json.dumps(source_config))
    output_config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "quant_type": "NVFP4",
        "group_size": 16,
        "ignore": sorted({*fallback_patterns, *auto_ignored_modules}),
        "swap_weight_nibbles": swap_weight_nibbles,
        "checkpoint_weight_scale_layout": "linear",
    }
    return output_config


def _build_direct_modelopt_nvfp4_transformer(
    *,
    base_dir: str,
    backbone_ckpt: str,
    output_path: Path,
    patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, int | bool]:
    source_config = _load_config(base_dir)
    state = _load_modelopt_backbone_state(backbone_ckpt)
    quantized_modules = _modelopt_nvfp4_modules(state)
    base_weight_map, index_filename = _load_weight_map(base_dir)
    if index_filename is None:
        raise ValueError("Direct ModelOpt NVFP4 export requires an indexed checkpoint")

    all_weight_modules = {
        name[: -len(".weight")] for name in base_weight_map if name.endswith(".weight")
    }
    fallback_modules = {
        module_name
        for module_name in all_weight_modules
        if _matches_any_pattern(module_name, patterns)
    }
    effective_modules = {
        name: tensors
        for name, tensors in quantized_modules.items()
        if name not in fallback_modules
    }
    auto_ignored_modules = sorted(all_weight_modules - set(effective_modules))
    output_config = _direct_nvfp4_output_config(
        source_config,
        fallback_patterns=patterns,
        auto_ignored_modules=auto_ignored_modules,
        swap_weight_nibbles=swap_weight_nibbles,
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    _copy_non_shard_files(base_dir, str(output_path))
    _write_config(output_path, output_config)
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in base_weight_map.items():
        names_by_file[filename].append(name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    added_scale_count = 0
    for filename, names in sorted(names_by_file.items()):
        shard_path = os.path.join(base_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in names:
            if not name.endswith(".weight"):
                continue
            module_name = name[: -len(".weight")]
            tensors = effective_modules.get(module_name)
            if tensors is None:
                continue
            tensors = _maybe_restore_h3_qkv_checkpoint_layout(
                module_name, tensors, source_config
            )
            shard_tensors[name] = tensors["weight"]
            for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
                scale_name = f"{module_name}.{suffix}"
                shard_tensors[scale_name] = tensors[suffix]
                added_scale_count += 1

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)
        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": updated_weight_map},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return {
        "quantized_modules": len(effective_modules),
        "fallback_modules": len(fallback_modules),
        "auto_ignored_modules": len(auto_ignored_modules),
        "added_scale_tensors": added_scale_count,
        "output_shards": len(names_by_file),
        "swap_weight_nibbles": swap_weight_nibbles,
    }


def _build_offline_modelopt_nvfp4_transformer(
    *,
    base_dir: str,
    calibration_ckpt: str,
    output_path: Path,
    patterns: Sequence[str],
    swap_weight_nibbles: bool,
    quant_device: str,
    quant_chunk_rows: int,
) -> dict[str, int | bool]:
    source_config = _load_config(base_dir)
    state = _load_modelopt_backbone_state(calibration_ckpt)
    input_amax_by_module = _modelopt_input_amax_map(state)
    weight_double_scale_by_module = _modelopt_weight_double_scale_map(state)
    if not input_amax_by_module:
        raise ValueError(
            "ModelOpt calibration checkpoint contains no input_quantizer._amax tensors"
        )

    base_weight_map, index_filename = _load_weight_map(base_dir)
    if index_filename is None:
        raise ValueError("Offline ModelOpt NVFP4 export requires an indexed checkpoint")
    all_weight_modules = {
        name[: -len(".weight")] for name in base_weight_map if name.endswith(".weight")
    }
    fallback_modules = {
        module_name
        for module_name in all_weight_modules
        if _matches_any_pattern(module_name, patterns)
    }
    effective_modules = (
        set(input_amax_by_module) & all_weight_modules
    ) - fallback_modules
    auto_ignored_modules = sorted(all_weight_modules - effective_modules)
    output_config = _direct_nvfp4_output_config(
        source_config,
        fallback_patterns=patterns,
        auto_ignored_modules=auto_ignored_modules,
        swap_weight_nibbles=swap_weight_nibbles,
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    device = torch.device(quant_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--offline-quant-device cuda requires an available CUDA GPU")

    _copy_non_shard_files(base_dir, str(output_path))
    _write_config(output_path, output_config)
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in base_weight_map.items():
        names_by_file[filename].append(name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    added_scale_count = 0
    for filename, names in sorted(names_by_file.items()):
        shard_path = os.path.join(base_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in names:
            if not name.endswith(".weight"):
                continue
            module_name = name[: -len(".weight")]
            if module_name not in effective_modules:
                continue
            packed_weight, block_scale, double_scale = _quantize_nvfp4_weight(
                shard_tensors[name].to(device=device),
                chunk_rows=quant_chunk_rows,
                double_scale=weight_double_scale_by_module.get(module_name),
            )
            input_scale = input_amax_by_module[module_name].to(torch.float32) / (
                _E2M1_MAX * _E4M3_MAX
            )
            if not torch.isfinite(input_scale).all() or not (input_scale > 0).all():
                raise ValueError(f"Invalid calibrated input amax for {module_name}")
            shard_tensors[name] = packed_weight.cpu()
            shard_tensors[f"{module_name}.weight_scale"] = block_scale.cpu()
            shard_tensors[f"{module_name}.weight_scale_2"] = double_scale.cpu()
            shard_tensors[f"{module_name}.input_scale"] = input_scale.cpu().contiguous()
            added_scale_count += 3

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)
        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()
        del shard_tensors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": updated_weight_map},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return {
        "quantized_modules": len(effective_modules),
        "fallback_modules": len(fallback_modules),
        "auto_ignored_modules": len(auto_ignored_modules),
        "added_scale_tensors": added_scale_count,
        "output_shards": len(names_by_file),
        "swap_weight_nibbles": swap_weight_nibbles,
    }


def build_modelopt_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    modelopt_hf_dir: str | None,
    modelopt_backbone_ckpt: str | None = None,
    output_dir: str,
    pattern_preset: str = "none",
    keep_bf16_patterns: Sequence[str] | None = None,
    swap_weight_nibbles: bool | None = None,
    allow_unquantized_source: bool = False,
    offline_quant_device: str = "cpu",
    offline_quant_chunk_rows: int = 1024,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    base_dir = _resolve_transformer_dir(base_transformer_dir)

    patterns = _preset_patterns(pattern_preset)
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)

    resolved_swap_weight_nibbles = (
        swap_weight_nibbles if swap_weight_nibbles is not None else False
    )
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if modelopt_backbone_ckpt is not None:
        if modelopt_hf_dir is not None:
            raise ValueError(
                "Use either --modelopt-hf-dir or --modelopt-backbone-ckpt, not both."
            )
        if allow_unquantized_source:
            return _build_offline_modelopt_nvfp4_transformer(
                base_dir=base_dir,
                calibration_ckpt=modelopt_backbone_ckpt,
                output_path=output_path,
                patterns=patterns,
                swap_weight_nibbles=resolved_swap_weight_nibbles,
                quant_device=offline_quant_device,
                quant_chunk_rows=offline_quant_chunk_rows,
            )
        return _build_direct_modelopt_nvfp4_transformer(
            base_dir=base_dir,
            backbone_ckpt=modelopt_backbone_ckpt,
            output_path=output_path,
            patterns=patterns,
            swap_weight_nibbles=resolved_swap_weight_nibbles,
        )

    if modelopt_hf_dir is None:
        raise ValueError(
            "Either --modelopt-hf-dir or --modelopt-backbone-ckpt is required."
        )
    source_dir = _resolve_transformer_dir(modelopt_hf_dir)
    output_config = _updated_quant_config(
        _load_config(source_dir),
        fallback_patterns=patterns,
        swap_weight_nibbles=resolved_swap_weight_nibbles,
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    _copy_non_shard_files(source_dir, str(output_path))
    _write_config(output_path, output_config)

    source_weight_map, index_filename = _load_weight_map(source_dir)
    base_weight_map, _ = _load_weight_map(base_dir)

    fallback_tensor_names = sorted(
        name
        for name in base_weight_map
        if name in source_weight_map
        and _matches_any_pattern(_module_name_for_tensor(name), patterns)
    )
    fallback_tensors = _load_selected_tensors(
        base_dir,
        base_weight_map,
        fallback_tensor_names,
    )
    fallback_modules = {
        _module_name_for_tensor(tensor_name) for tensor_name in fallback_tensor_names
    }

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in source_weight_map.items():
        weights_by_file[filename].append(tensor_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    replaced_tensor_count = 0
    removed_aux_tensor_count = 0

    for filename, tensor_names in sorted(weights_by_file.items()):
        shard_path = os.path.join(source_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})

        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in list(shard_tensors.keys()):
            if "_quantizer." in name:
                del shard_tensors[name]
                removed_aux_tensor_count += 1
                continue

            module_name = _module_name_for_tensor(name)
            if module_name not in fallback_modules:
                continue

            if name in fallback_tensors:
                shard_tensors[name] = fallback_tensors[name]
                replaced_tensor_count += 1
            else:
                del shard_tensors[name]
                removed_aux_tensor_count += 1

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)

        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

    if index_filename is None:
        raise ValueError(
            "Expected a sharded or indexed ModelOpt HF export, but no index file was found."
        )

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": updated_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return {
        "fallback_modules": len(fallback_modules),
        "replaced_tensors": replaced_tensor_count,
        "removed_aux_tensors": removed_aux_tensor_count,
        "output_shards": len(weights_by_file),
        "swap_weight_nibbles": resolved_swap_weight_nibbles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer and "
            "optionally keep selected modules in BF16."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help="Original BF16 transformer directory, or a parent model directory.",
    )
    parser.add_argument(
        "--modelopt-hf-dir",
        default=None,
        help="ModelOpt --hf-ckpt-dir output, or its transformer subdirectory.",
    )
    parser.add_argument(
        "--modelopt-backbone-ckpt",
        default=None,
        help=(
            "Compressed ModelOpt mto.save checkpoint. This native H3 path uses "
            "the original BF16 transformer as the source and does not require an "
            "HF export."
        ),
    )
    parser.add_argument(
        "--allow-unquantized-source",
        action="store_true",
        help=(
            "Treat --modelopt-backbone-ckpt as a scalar calibration state "
            "(input amax and optional exact weight double scale) and quantize "
            "original BF16 weights one shard at a time."
        ),
    )
    parser.add_argument(
        "--offline-quant-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for --allow-unquantized-source weight packing.",
    )
    parser.add_argument(
        "--offline-quant-chunk-rows",
        type=int,
        default=1024,
        help="Maximum rows quantized at once to bound FP4 packing memory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the mixed transformer checkpoint.",
    )
    parser.add_argument(
        "--pattern-preset",
        choices=["none", "flux1-nvfp4", "minimax-h3-nvfp4"],
        default="none",
        help="Optional model-family BF16 fallback preset.",
    )
    parser.add_argument(
        "--keep-bf16-pattern",
        action="append",
        default=[],
        help=(
            "Glob-style pattern matched against module names without trailing tensor "
            "suffixes such as .weight or .bias."
        ),
    )
    parser.add_argument(
        "--swap-weight-nibbles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether the runtime should swap packed FP4 nibbles before padding. "
            "Defaults to false."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=args.base_transformer_dir,
        modelopt_hf_dir=args.modelopt_hf_dir,
        modelopt_backbone_ckpt=args.modelopt_backbone_ckpt,
        output_dir=args.output_dir,
        pattern_preset=args.pattern_preset,
        keep_bf16_patterns=args.keep_bf16_pattern,
        swap_weight_nibbles=args.swap_weight_nibbles,
        allow_unquantized_source=args.allow_unquantized_source,
        offline_quant_device=args.offline_quant_device,
        offline_quant_chunk_rows=args.offline_quant_chunk_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
