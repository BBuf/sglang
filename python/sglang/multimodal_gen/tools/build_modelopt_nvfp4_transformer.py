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
"""

from __future__ import annotations

import argparse
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


def build_modelopt_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    modelopt_hf_dir: str | None,
    modelopt_backbone_ckpt: str | None = None,
    output_dir: str,
    pattern_preset: str = "none",
    keep_bf16_patterns: Sequence[str] | None = None,
    swap_weight_nibbles: bool | None = None,
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
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
