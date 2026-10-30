def _joy_image_mlp(mlp: MLP, x: torch.Tensor, fuse_gelu: bool) -> torch.Tensor:
    """Use the SM120 image epilogue when eligible; retain the original MLP otherwise."""
    fc_in = mlp.fc_in
    # SM120's epilogue wins for the large image projection, but regresses the
    # smaller text projection. It also removes the intermediate BF16 rounding.
    # Native single-reference image buckets have at least 7168 image tokens.
    if (
        not fuse_gelu
        or torch.compiler.is_compiling()
        or not x.is_cuda
        or torch.version.hip
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] < 7168
        or x.shape[2] != 4096
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (12, 0)
        or not isinstance(mlp.act, nn.GELU)
        or mlp.act.approximate != "tanh"
        or (isinstance(fc_in, BaseLayerWithLoRA) and fc_in.lora_active)
        or not can_use_linear_gelu_static(fc_in)
        or fc_in.weight.shape != (16384, 4096)
        or fc_in.weight.dtype != x.dtype
        or fc_in.weight.device != x.device
        or fc_in.bias.device != x.device
    ):
        return mlp(x)
    hidden = fused_linear_gelu_tanh(x, fc_in.weight, fc_in.bias)
    # Keep the output projection wrapper, including LoRA and TP reduction.
    return mlp.fc_out(hidden)[0]
