def _joy_image_mlp(mlp: MLP, x: torch.Tensor, fuse_gelu: bool) -> torch.Tensor:
    linear = mlp.fc_in
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
        or (isinstance(linear, BaseLayerWithLoRA) and linear.lora_active)
        or not can_use_linear_gelu_static(linear)
        or linear.weight.shape != (16384, 4096)
        or linear.weight.dtype != x.dtype
        or linear.weight.device != x.device
        or linear.bias.device != x.device
    ):
        return mlp(x)
    x = fused_linear_gelu_tanh(x, linear.weight, linear.bias)
    return mlp.fc_out(x)[0]
