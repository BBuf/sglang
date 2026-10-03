"""Same-source, same-input baseline/candidate benchmark for the cleanup."""
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    residual_gate_add, modulate_scale_shift, usp_merge_heads,
    fused_causal_conv3d_cat_pad_cuda,
)
from sglang.multimodal_gen.runtime.models.dits.helios import HeliosSelfAttention


def rand(shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype)


def original_workloads():
    for label, shape, gate_shape, transposed in [
        ("flux_text", (1, 512, 3072), (1, 1, 3072), False),
        ("flux_image", (1, 4096, 3072), (1, 1, 3072), False),
        ("flux2_joint", (1, 4608, 6144), (1, 1, 6144), False),
        ("full_gate", (1, 512, 4096), (1, 512, 4096), False),
        ("per_token", (1, 2560, 512), (1, 2560, 1), False),
        ("odd_width", (2, 33, 65), (1, 1, 65), False),
        ("sana_transposed", (1, 32760, 2240), (1, 1, 2240), True),
        ("transposed_small", (1, 17, 65), (1, 1, 65), True),
        ("transposed_medium", (2, 513, 256), (1, 1, 256), True),
    ]:
        for dtype in (torch.bfloat16, torch.float16):
            r = rand((shape[0], shape[2], shape[1]), dtype).transpose(1, 2) if transposed else rand(shape, dtype)
            yield f"residual/{label}/{dtype}", residual_gate_add, (r, rand(shape, dtype), rand(gate_shape, dtype)), None
    for shape in ((1, 512, 3072), (1, 4096, 3072), (2, 1024, 3072)):
        yield f"modulate/{shape}", modulate_scale_shift, (rand(shape), rand((shape[0], shape[-1])), rand((shape[0], shape[-1]))), None
    for shape in ((4, 7936, 1, 14, 128), (2, 64, 3, 4, 64), (4, 33, 2, 4, 100)):
        yield f"usp/{shape}", usp_merge_heads, (rand(shape),), None
    attention = HeliosSelfAttention.__new__(HeliosSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.tp_rmsnorm = False
    for shape in ((1, 2160, 40, 128), (1, 8640, 40, 128), (2, 17, 8, 128)):
        freqs = rand((shape[0], shape[1], 256), torch.float32) * 0.01
        yield f"helios/{shape}", attention._apply_rotary_qk, (rand(shape), rand(shape), freqs), None
    for c, h, w in ((8, 6, 6), (512, 30, 52), (1024, 30, 52)):
        yield f"cat_pad/{c}/{h}/{w}", fused_causal_conv3d_cat_pad_cuda, (rand((1,c,1,h,w)), rand((1,c,1,h,w)), (1,1,1,1,1,0)), None


def workloads():
    from sglang.kernels.ops import diffusion as ops
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import qknorm_complex_rope
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import qknorm_complex_rope_kv
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21
    from sglang.srt.layers.layernorm import RMSNorm
    yield from original_workloads()
    for seq in (17, 4096):
        shape=(1, seq, 32, 128)
        q,k=rand(shape),rand(shape)
        rope=torch.polar(torch.ones((seq,64),device="cuda"),rand((seq,64),torch.float32))
        weight=rand((128,))
        norm=RMSNorm(128,1e-6,cast_x_before_out_mul=True,force_native=True).to(device=q.device,dtype=q.dtype)
        norm.weight.data.copy_(weight)
        def model_qk(x, cache):
            return qwen_image21.apply_qk_norm_rope(x, norm, cache)
        yield f"qwen_model_contiguous/{seq}",model_qk,(q,rope),None
        packed=rand((1,seq,3,32,128))
        def model_packed(parent, cache):
            return model_qk(parent.unbind(2)[0], cache)
        yield f"qwen_model_packed/{seq}",model_packed,(packed,rope),None
        yield f"complex_rope/{seq}",ops.fused_complex_rope,(q,rope),None
        yield f"qknorm_complex_triton/{seq}",qknorm_complex_rope,(q,weight,rope,1e-6),None
        yield f"qknorm_complex_cuda/{seq}",ops.qknorm_complex_rope_cuda,(q,weight,rope,1e-6),None
        yield f"qknorm_complex_kv/{seq}",qknorm_complex_rope_kv,(k,weight,rope,rand(shape),rand((1,18,32,128)),rand((1,18,32,128)),1e-6),None
        yield f"rmsnorm_preserve/{seq}",ops.rmsnorm_preserve_reduction,(q,weight,1e-6),None
        cos,sin=rand((1,seq,1,128),torch.float64),rand((1,seq,1,128),torch.float64)
        yield f"interleaved_fp64/{seq}",ops.fused_interleaved_rope_fp64,(q,k,cos,sin),None
        yield f"rotate_half/{seq}",ops.fused_rope_rotate_half_bitexact,(q,rand((seq,128)),rand((seq,128))),None
        yield f"qk_layernorm/{seq}",ops.fused_qk_head_layernorm,(q,k,1e-6),None
        x=rand((1,seq,4096));row=rand((1,4096));vec=rand((1,1,4096));w=rand((4096,))
        yield f"layernorm_modulate/{seq}",ops.fused_layernorm_modulate_raw,(x,row,row,1e-6),None
        yield f"rmsnorm_modulate/{seq}",ops.fused_rmsnorm_scale_shift_bitexact,(x,w,vec,vec,1e-6),None
        yield f"residual_rmsnorm/{seq}",ops.fused_scale_residual_rmsnorm_scale_shift_bitexact,(x,rand(x.shape),vec,w,vec,vec,1e-6),None
        yield f"gelu_cat/{seq}",ops.fused_gelu_tanh_cat,(x,rand((1,seq,16384))),None
        yield f"silu_mul/{seq}",ops.fused_silu_mul_bitexact,(x,rand(x.shape)),None
        yield f"wan_temb/{seq}",ops.fused_temb_table_slices,(rand((1,6,512),torch.float32),rand((1,seq,6,512))),None
    for frames,height,width,heads in ((3,7,7,1),(18,17,30,32)):
        dims=(16,24,24)
        tables=tuple((rand((n,d//2),torch.float32),rand((n,d//2),torch.float32)) for n,d in zip((frames,height,width),dims))
        q=rand((1,frames,height,width,heads,64));k=rand(q.shape)
        yield f"ltx25/{frames}/{height}/{width}",ops.fused_ltx25_decoder_rope,(q,k,*(t for pair in tables for t in pair),dims[0],dims[1]),None
    for layout in (torch.contiguous_format,torch.channels_last):
        x=rand((2,2240,22,40)).to(memory_format=layout);bias=rand((2240,))
        yield f"sana_silu/{layout}",ops.fused_bias_silu,(x,bias),None
        yield f"sana_glu/{layout}",ops.fused_bias_glu,(x,bias),None
    for seq in (17,4096):
        image=tuple(rand((1,seq,32,128)) for _ in range(3))
        text=tuple(rand((1,18,32,128)) for _ in range(3))
        yield f"joint_qkv/{seq}",ops.joint_qkv_cat,(*image,*text),None
    for frames,heads in ((1,1),(11,7)):
        k=rand((frames,heads,128,32),torch.float32)*0.01
        A=k@k.transpose(-1,-2);B=rand(A.shape,torch.float32);alpha=rand((frames,heads,128),torch.float32)
        yield f"vdn/{frames}/{heads}",ops.vdn_delta_factors,(A,B,alpha),None


def fingerprints(out):
    if isinstance(out, torch.Tensor):
        raw=out.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return dict(shape=list(out.shape), stride=list(out.stride()), sha256=hashlib.sha256(raw).hexdigest())
    return [fingerprints(t) for t in out]


@torch.inference_mode()
def main():
    torch.manual_seed(20261003)
    result = dict(torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(0), arm=os.environ['ARM'], cases=[])
    for name, fn, args, kwargs in workloads():
        out = fn(*args)
        fingerprint = fingerprints(out)
        del out
        graph = marker.do_bench(fn, input_args=args, warmup_iters=10, replay_iters=100,
                               memory_output=None, disable_log_bandwidth=True)
        host = []
        for _ in range(3):
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(100): fn(*args)
            torch.cuda.synchronize()
            host.append((time.perf_counter()-start)*1e4)
        row=dict(name=name, graph_us=graph.times[0]*1e6, eager_us=statistics.median(host), output=fingerprint)
        result['cases'].append(row)
        print(json.dumps(row), flush=True)
    Path(os.environ['OUTPUT']).write_text(json.dumps(result, indent=2)+'\n')

if __name__=='__main__':
    main()
