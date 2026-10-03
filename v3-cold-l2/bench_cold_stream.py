"""Per-invocation cold-L2 graph timing; eviction is outside the event interval."""
import gc
import json
import os
import statistics
from pathlib import Path

import torch
import triton

from bench_workloads import fingerprints, rand, workloads


def clone_args(args):
    copies = {}
    def clone(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in copies:
                copies[id(value)] = value.clone(memory_format=torch.preserve_format)
            return copies[id(value)]
        if isinstance(value, tuple):
            return tuple(clone(x) for x in value)
        return value
    return clone(args)


def extra_workloads():
    from sglang.kernels.ops import diffusion as ops
    from sglang.multimodal_gen.runtime.models.dits import glm_image, ernie_image, wanvideo
    from sglang.srt.layers.layernorm import RMSNorm
    # A CPU request must not poison the subsequent CUDA fusion gate.
    wanvideo._wan_temb_table_slices(torch.zeros(1, 6, 64), torch.zeros(1, 9, 6, 64))
    assert not wanvideo._WAN_TEMB_SLICES.disabled
    for seq in (17, 4096):
        x = rand((1, seq, 4096)); row = rand((1, 4096)); vec = row[:, None]
        q, k = rand((1, seq, 32, 128)), rand((1, seq, 32, 128))
        ln = torch.nn.LayerNorm(4096, elementwise_affine=False, eps=1e-6).cuda()
        qln = torch.nn.LayerNorm(128, elementwise_affine=False, eps=1e-6).cuda()
        norm = RMSNorm(4096, eps=1e-6).to(device=x.device, dtype=x.dtype)
        def glm_ln(x, scale, shift):
            return glm_image._glm_ln_modulate(ln, x, scale, shift, x.dtype)
        def glm_qk(q, k):
            return glm_image._glm_qk_layernorm(qln, qln, q, k, q.dtype)
        def ernie_rms(x, scale, shift):
            return ernie_image._ernie_norm_scale_shift(norm, x, scale, shift)
        yield f'model_glm_ln/{seq}', glm_ln, (x, row, row), None
        yield f'model_glm_qk/{seq}', glm_qk, (q, k), None
        yield f'model_ernie_rms/{seq}', ernie_rms, (x, vec, vec), None
        yield f'model_wan_slices/{seq}', wanvideo._wan_temb_table_slices, (rand((1, 6, 512), torch.float32), rand((1, seq, 6, 512))), None
        rope = torch.polar(torch.ones((seq, 64), device='cuda'), rand((seq, 64), torch.float32))
        weight = rand((128,))
        for packed in (False, True):
            if packed:
                parent = rand((1, seq + 3, 3, 32, 128))
                kp, vp = rand((1, 3, 32, 128)), rand((1, 3, 32, 128))
                def pack_views(parent, weight, rope, kp, vp):
                    q = parent[:, 3:, 0]
                    k_out, v_out = parent[:, :, 1], parent[:, :, 2]
                    ops.qknorm_complex_rope_pack_(q, k_out, v_out, weight, weight, rope, kp, vp, None, None, 1e-6)
                    return q, k_out, v_out
                yield f'qknorm_pack_strided/{seq}', pack_views, (parent, weight, rope, kp, vp), None
            else:
                k_out, v_out = rand((1, seq + 3, 32, 128)), rand((1, seq + 3, 32, 128))
                kp, vp = rand((1, 3, 32, 128)), rand((1, 3, 32, 128))
                def pack_dense(q, k, v, k_out, v_out, weight, rope, kp, vp):
                    ops.qknorm_complex_rope_pack_(q, k_out, v_out, weight, weight, rope, kp, vp, k, v, 1e-6)
                    return q, k_out, v_out
                yield f'qknorm_pack_dense/{seq}', pack_dense, (q, k, rand(q.shape), k_out, v_out, weight, rope, kp, vp), None


def cold_bench(fn, args, eviction, expected):
    static = clone_args(args)
    for _ in range(10):
        fn(*static)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(100):
        for dst, src in zip(static, args):
            if isinstance(src, torch.Tensor):
                dst.copy_(src)
        # Give the CPU time to enqueue the entire timed interval before GPU
        # execution. Both this delay and the L2 eviction precede start.
        torch.cuda._sleep(10_000_000)
        eviction.add_(1)
        start.record()
        output = fn(*static)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    assert fingerprints(output) == expected
    return dict(cold_median_us=statistics.median(samples), cold_min_us=min(samples),
                cold_p95_us=sorted(samples)[94], samples_us=samples,
                replay_output_equal=True)


@torch.inference_mode()
def main():
    torch.manual_seed(20261004)
    props = torch.cuda.get_device_properties(0)
    eviction = torch.zeros(props.L2_cache_size * 5, device='cuda', dtype=torch.uint8)
    result = dict(arm=os.environ['ARM'], torch=torch.__version__, triton=triton.__version__,
                  gpu=props.name, l2_bytes=props.L2_cache_size, eviction_bytes=eviction.numel(),
                  method='stream: reset inputs; GPU delay; read/write 5x L2; ordinary start event; one queued operator; ordinary end event',
                  measured_calls=100, eager_warmups=10, cases=[])
    # Control quantifies the event-node interval; never subtract it from results.
    result['empty_interval'] = cold_bench(lambda x: x, (torch.zeros(1, device='cuda'),), eviction,
                                         fingerprints(torch.zeros(1, device='cuda')))
    def all_workloads():
        yield from workloads()
        yield from extra_workloads()
    pattern = os.environ.get('FILTER', '')
    for name, fn, args, _ in all_workloads():
        if pattern and not any(p in name for p in pattern.split(',')):
            continue
        output = fn(*clone_args(args))
        expected = fingerprints(output)
        del output
        row = dict(name=name, output=expected, **cold_bench(fn, args, eviction, expected))
        result['cases'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}), flush=True)
        gc.collect()
    Path(os.environ['OUTPUT']).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
