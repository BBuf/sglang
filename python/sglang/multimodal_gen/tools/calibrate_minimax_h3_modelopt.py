"""Calibrate the native MiniMax H3 DiT with NVIDIA ModelOpt.

MiniMax H3 is released as a native SGLang transformer rather than a Diffusers
``ModelMixin``.  ModelOpt therefore cannot discover its tensor-parallel linear
classes through the stock Diffusers calibration script.  This tool registers
those classes with ModelOpt, loads only the H3 transformer, calibrates it on a
small packed audio/video workload, and writes an ``mto.save`` checkpoint for
the existing SGLang ModelOpt checkpoint builders.

The built-in synthetic workload is a smoke/calibration fallback.  Promotion of
an exported checkpoint still requires same-prompt, same-seed end-to-end quality
validation against the BF16 pipeline.

Example with kwargs captured from a representative native H3 denoising run::

    SGLANG_MINIMAX_H3_MODELOPT_CAPTURE_DIR=/tmp/h3-calibration \
    SGLANG_MINIMAX_H3_MODELOPT_CAPTURE_MAX_SAMPLES=8 \
        sglang generate --config /path/to/h3-generation.json

    python -m sglang.multimodal_gen.tools.calibrate_minimax_h3_modelopt \
        --transformer-dir /path/to/MiniMax-H3/FL2VA/transformer \
        --calibration-data /tmp/h3-calibration \
        --format fp8 \
        --output-checkpoint /tmp/minimax-h3-fp8/backbone.pt

The default quality profile leaves all AdaLN projections, the text refiner,
the first/last attention blocks, and fixed-FP32 input/output boundaries
unquantized.  Pass ``--sensitive-layer-pattern`` to replace that profile.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

REQUIRED_UNQUANTIZED_LAYER_PATTERNS = (
    "video_patch_proj",
    "audio_patch_proj",
    "time_embedder",
    "final_layer.video_out",
    "final_layer.audio_out",
)
DEFAULT_SENSITIVE_LAYER_PATTERNS = REQUIRED_UNQUANTIZED_LAYER_PATTERNS + (
    "condition_proj",
    "token_refiner",
    "blocks.0.attn",
    "blocks.*.adaln_proj",
    "blocks.49.attn",
    "final_layer.adaln_proj",
)


def _resolve_transformer_dir(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return candidate
    for relative in ("FL2VA/transformer", "transformer"):
        transformer_dir = candidate / relative
        if (transformer_dir / "config.json").is_file():
            return transformer_dir
    raise FileNotFoundError(f"Could not resolve a MiniMax H3 transformer from {path}")


def _model_root_from_transformer_dir(transformer_dir: Path) -> Path:
    if transformer_dir.parent.name.lower() in {"fl2va", "ref2va"}:
        # A selectively downloaded native H3 checkpoint still carries a complete
        # model_index.json inside the variant directory.  Point ServerArgs there
        # directly; the modular repository root only has modular_model_index.json.
        return transformer_dir.parent
    return transformer_dir.parent


def _initialize_single_gpu_parallel_runtime() -> None:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29673")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _load_transformer(transformer_dir: Path) -> torch.nn.Module:
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    model_root = _model_root_from_transformer_dir(transformer_dir)
    server_args = ServerArgs.from_kwargs(
        model_path=str(model_root),
        num_gpus=1,
        tp_size=1,
        ulysses_degree=1,
        performance_mode="speed",
        enable_torch_compile=False,
    )
    set_global_server_args(server_args)
    return TransformerLoader().load_customized(
        str(transformer_dir), server_args, "transformer"
    )


def _register_sglang_linears_with_modelopt() -> None:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear

    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    for linear_cls in (
        ReplicatedLinear,
        ColumnParallelLinear,
        RowParallelLinear,
        MergedColumnParallelLinear,
    ):
        try:
            mtq.unregister(linear_cls)
        except (KeyError, RuntimeError, ValueError):
            pass
        mtq.register(linear_cls, _QuantLinear)


def _modelopt_default_config(quant_format: str) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq

    config_names = {
        "fp8": ("FP8_DEFAULT_CFG", "FP8_DEFAULT_CONFIG"),
        "nvfp4": ("NVFP4_DEFAULT_CFG", "NVFP4_DEFAULT_CONFIG"),
    }
    for name in config_names[quant_format]:
        config = getattr(mtq, name, None)
        if config is not None:
            return copy.deepcopy(config)
    raise RuntimeError(
        f"Installed ModelOpt does not provide a {quant_format} default config"
    )


def _disable_sensitive_layers(
    quant_config: dict[str, Any], patterns: Sequence[str]
) -> None:
    raw_quant_cfg = quant_config.get("quant_cfg", {})
    if isinstance(raw_quant_cfg, dict):
        # ModelOpt <= 0.43 uses an ordered wildcard dictionary.  A module-level
        # pattern disables every quantizer below the matched linear.
        quant_cfg = dict(raw_quant_cfg)
        for pattern in patterns:
            quant_cfg[f"*{pattern}*"] = {"enable": False}
    elif isinstance(raw_quant_cfg, list):
        # Newer ModelOpt releases use explicit per-quantizer list entries.
        quant_cfg = list(raw_quant_cfg)
        for pattern in patterns:
            for quantizer in ("weight_quantizer", "input_quantizer"):
                quant_cfg.append(
                    {
                        "quantizer_name": f"*{pattern}*{quantizer}",
                        "enable": False,
                    }
                )
    else:
        raise TypeError(
            "ModelOpt quant_cfg must be an ordered mapping or list, got "
            f"{type(raw_quant_cfg).__name__}"
        )
    quant_config["quant_cfg"] = quant_cfg


def _split_calibration_rows(seq_len: int) -> tuple[int, int, int]:
    if seq_len < 64 or seq_len % 64:
        raise ValueError("--calibration-seq-len must be a multiple of 64 and >= 64")
    text_rows = max(16, seq_len // 4)
    audio_rows = max(16, seq_len // 4)
    video_rows = seq_len - text_rows - audio_rows
    return text_rows, video_rows, audio_rows


def _synthetic_calibration_kwargs(
    model: torch.nn.Module,
    *,
    seq_len: int,
    timesteps: Iterable[float],
    seed: int,
) -> list[dict[str, Any]]:
    arch = model.arch
    text_rows, video_rows, audio_rows = _split_calibration_rows(seq_len)
    text_stop = text_rows
    video_stop = text_stop + video_rows
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    text_pos = torch.arange(0, text_stop, device=device, dtype=torch.long)
    img_pos = torch.arange(text_stop, video_stop, device=device, dtype=torch.long)
    audio_pos = torch.arange(video_stop, seq_len, device=device, dtype=torch.long)
    token_tags = torch.empty(seq_len, device=device, dtype=torch.long)
    token_tags[:text_stop] = 0
    token_tags[text_stop:video_stop] = 1
    token_tags[video_stop:] = 2

    packed_position_ids = torch.zeros(
        (1, seq_len, 3), device=device, dtype=torch.float32
    )
    packed_position_ids[0, :, 0] = torch.arange(
        seq_len, device=device, dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    refiner_cu = torch.tensor([0, text_rows], device=device, dtype=torch.int32)
    inverse_indices = torch.zeros(seq_len, device=device, dtype=torch.long)
    common = {
        "img_position_ids": packed_position_ids,
        "inverse_indices": inverse_indices,
        "update_mask": torch.ones(video_rows, device=device, dtype=torch.float32),
        "update_audio_mask": torch.ones(audio_rows, device=device, dtype=torch.float32),
        "token_tags": token_tags,
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": img_pos},
        "packed_seq_params": {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_q_host": (0, seq_len),
            "max_seqlen_q": seq_len,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": refiner_cu,
            "max_seqlen_q": text_rows,
        },
    }

    samples = []
    for timestep in timesteps:
        samples.append(
            {
                **common,
                "x": torch.randn(
                    (1, seq_len, arch.latents_dim * math.prod(arch.patch_size)),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                ),
                "audio_x": torch.randn(
                    (1, seq_len, arch.audio_latents_dim),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                ),
                "prompt_embeds": torch.randn(
                    (text_rows, arch.text_dim),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                ),
                "unique_timesteps": torch.tensor(
                    [timestep], device=device, dtype=torch.float32
                ),
            }
        )
    return samples


def _move_calibration_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _move_calibration_value(item, device) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_calibration_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_calibration_value(item, device) for item in value]
    return value


def _load_captured_calibration_kwargs(
    path: str, model: torch.nn.Module
) -> list[dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    files = sorted(candidate.glob("*.pt")) if candidate.is_dir() else [candidate]
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(f"No captured calibration .pt files found at {path}")

    samples: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "samples" in payload:
            payloads = payload["samples"]
        else:
            payloads = payload if isinstance(payload, list) else [payload]
        for raw_sample in payloads:
            if isinstance(raw_sample, dict) and "kwargs" in raw_sample:
                raw_sample = raw_sample["kwargs"]
            if not isinstance(raw_sample, dict):
                raise TypeError(
                    f"Captured calibration sample in {file} must be a kwargs dict"
                )
            sample = dict(raw_sample)
            # Serving may cache device-specific RoPE tensors and Ulysses row maps.
            # Recompute those for this single-GPU calibration process.
            sample.pop("rope_cache", None)
            sample.pop("local_embedding_layout", None)
            samples.append(_move_calibration_value(sample, device))
    if not samples:
        raise ValueError(f"Captured calibration data at {path} contains no samples")
    return samples


def calibrate_minimax_h3(
    *,
    transformer_dir: str,
    output_checkpoint: str,
    quant_format: str,
    sensitive_layer_patterns: Sequence[str],
    calibration_seq_len: int,
    calibration_timesteps: Sequence[float],
    seed: int,
    calibration_data: str | None,
    allow_synthetic_calibration: bool,
) -> dict[str, Any]:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    resolved_transformer_dir = _resolve_transformer_dir(transformer_dir)
    _initialize_single_gpu_parallel_runtime()
    model = _load_transformer(resolved_transformer_dir).eval()
    _register_sglang_linears_with_modelopt()
    quant_config = _modelopt_default_config(quant_format)
    effective_sensitive_patterns = tuple(
        dict.fromkeys((*REQUIRED_UNQUANTIZED_LAYER_PATTERNS, *sensitive_layer_patterns))
    )
    _disable_sensitive_layers(quant_config, effective_sensitive_patterns)
    if calibration_data is not None:
        calibration_samples = _load_captured_calibration_kwargs(calibration_data, model)
        calibration_source = str(Path(calibration_data).expanduser().resolve())
    else:
        if not allow_synthetic_calibration:
            raise ValueError(
                "Real MiniMax H3 denoiser captures are required for a promotable "
                "checkpoint. Pass --calibration-data, or explicitly use "
                "--allow-synthetic-calibration for smoke testing only."
            )
        calibration_samples = _synthetic_calibration_kwargs(
            model,
            seq_len=calibration_seq_len,
            timesteps=calibration_timesteps,
            seed=seed,
        )
        calibration_source = "synthetic-smoke-only"

    def _forward_loop(_: torch.nn.Module) -> None:
        for sample in calibration_samples:
            video, audio = model(**sample)
            if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                raise RuntimeError("MiniMax H3 calibration produced non-finite logits")

    mtq.quantize(model, quant_config, _forward_loop)
    if quant_format == "nvfp4":
        mtq.compress(model)

    output_path = Path(output_checkpoint).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mto.save(model, str(output_path))
    metadata = {
        "format": quant_format,
        "source_transformer": str(resolved_transformer_dir),
        "sensitive_layer_patterns": list(effective_sensitive_patterns),
        "calibration_source": calibration_source,
        "calibration_sample_count": len(calibration_samples),
        "seed": seed,
    }
    if calibration_data is None:
        metadata["synthetic_calibration"] = {
            "seq_len": calibration_seq_len,
            "timesteps": list(calibration_timesteps),
        }
    with open(output_path.with_suffix(output_path.suffix + ".json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--format", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument(
        "--sensitive-layer-pattern",
        action="append",
        default=None,
        help=(
            "ModelOpt module-name substring to keep unquantized. May be repeated. "
            "Defaults to the conservative MiniMax H3 profile."
        ),
    )
    parser.add_argument("--calibration-seq-len", type=int, default=128)
    parser.add_argument(
        "--calibration-data",
        help=(
            "Captured native H3 forward kwargs (.pt file or directory of .pt "
            "files) from a representative denoising trajectory."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-calibration",
        action="store_true",
        help=(
            "Use random inputs when --calibration-data is absent. Intended only "
            "for loader smoke tests; random H3 activations are not promotable."
        ),
    )
    parser.add_argument(
        "--calibration-timesteps",
        type=float,
        nargs="+",
        default=(1000.0, 750.0, 500.0, 250.0, 50.0),
    )
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patterns = (
        tuple(args.sensitive_layer_pattern)
        if args.sensitive_layer_pattern is not None
        else DEFAULT_SENSITIVE_LAYER_PATTERNS
    )
    try:
        metadata = calibrate_minimax_h3(
            transformer_dir=args.transformer_dir,
            output_checkpoint=args.output_checkpoint,
            quant_format=args.format,
            sensitive_layer_patterns=patterns,
            calibration_seq_len=args.calibration_seq_len,
            calibration_timesteps=args.calibration_timesteps,
            seed=args.seed,
            calibration_data=args.calibration_data,
            allow_synthetic_calibration=args.allow_synthetic_calibration,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
