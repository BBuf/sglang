"""Diagnostic-only real-call signatures; never enabled during timing rows."""
import atexit
import collections
import functools
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path

if os.environ.get('KDA_CAPTURE_DIR'):
    output = Path(os.environ['KDA_CAPTURE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    targets = {
        'norm.group_norm_silu_twopass_triton': ['group_norm_silu_4d', 'group_norm_silu_rows'],
        'norm.group_norm_silu_triton': ['triton_group_norm_silu'],
        'norm.native_bf16_rmsnorm_triton': ['rmsnorm_scale', 'rmsnorm_tanh_residual'],
        'layout.wan_causal_cache_triton': ['cat_pad_channels_last_3d'],
        'layout.nearest_upsample_nhwc_triton': ['nearest_upsample_nhwc'],
        'modulate.wan_temb_table_slices_triton': ['fused_temb_table_slices'],
        'activation.silu_mul_bitexact': ['fused_silu_mul_bitexact', 'fused_packed_silu_mul_bitexact'],
        'rope.hunyuan_qkv_pack_triton': ['hunyuan_qkv_rope_pack'],
        'rope.complex_rope_triton': ['fused_complex_rope'],
        'rope.rope_rotate_half_bitexact': ['fused_rope_rotate_half_bitexact'],
        'layout.joint_qkv_cat_triton': ['joint_qkv_cat'],
        'activation.sana_conv_post_triton': ['fused_bias_silu', 'fused_bias_glu'],
        'modulate.ltx2_ada_values_triton': ['ltx2_ada_values9'],
    }
    def describe(value):
        if hasattr(value, 'shape') and hasattr(value, 'stride'):
            return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device), storage_offset=value.storage_offset())
        if isinstance(value, (tuple, list)):
            return [describe(x) for x in value]
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        return value if isinstance(value, (str, int, float, bool, type(None))) else str(type(value))
    def instrument(fn, qualified):
        shadow = None
        if qualified.endswith('.ltx2_ada_values9') and os.environ.get('KDA_SHADOW_BASELINE'):
            source = Path(os.environ['KDA_SHADOW_BASELINE']) / 'python/sglang/kernels/ops/diffusion/modulate/ltx2_ada_values_triton.py'
            spec = importlib.util.spec_from_file_location('kda_shadow_baseline_ltx2_ada', source)
            reference = importlib.util.module_from_spec(spec)
            import sys
            sys.modules[spec.name] = reference
            spec.loader.exec_module(reference)
            shadow = reference.ltx2_ada_values9
            (output / f'shadow-source-{os.getpid()}.json').write_text(json.dumps(dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if shadow is not None:
                import torch
                inputs = list(args) + list(kwargs.values())
                snapshots = [x.detach().clone() for x in inputs]
            result = fn(*args, **kwargs)
            row = dict(function=qualified, args=describe(args), kwargs=describe(kwargs), result=describe(result))
            if shadow is not None:
                expected = shadow(*args, **kwargs)
                def exact(a, b):
                    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                def aliases(values):
                    return [[x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr() for y in values] for x in values]
                check = dict(function=qualified, args=describe(args),
                             output_exact=[exact(a, b) for a, b in zip(result, expected)],
                             metadata_exact=describe(result) == describe(expected),
                             alias_pattern_exact=aliases(list(result) + inputs) == aliases(list(expected) + inputs),
                             inputs_preserved=all(exact(a, b) for a, b in zip(inputs, snapshots)))
                check['qualified'] = all(check['output_exact']) and check['metadata_exact'] and check['alias_pattern_exact'] and check['inputs_preserved']
                with (output / f'shadow-{os.getpid()}.jsonl').open('a') as f:
                    f.write(json.dumps(check, sort_keys=True) + '\n')
                assert check['qualified'], f'Live-input LTX2 ada9 baseline/candidate mismatch: {check}'
            with (output / f'calls-{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(row, sort_keys=True) + '\n')
            return result
        return wrapper
    for suffix, names in targets.items():
        module_name = 'sglang.kernels.ops.diffusion.' + suffix
        try:
            module = importlib.import_module(module_name)
            for name in names:
                setattr(module, name, instrument(getattr(module, name), module_name + '.' + name))
        except Exception as exc:
            with (output / f'install-{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(dict(module=module_name, error=repr(exc))) + '\n')
            if os.environ.get('KDA_SHADOW_BASELINE') and suffix == 'modulate.ltx2_ada_values_triton':
                raise
