"""Standalone numerical feasibility probe; never edits or loads a model.

Run only after the final LN driver and archive have terminated. A passing random
probe is not model admission: real activations, final outputs and paired serving
measurements remain mandatory. Every attempted accumulation order is retained.
"""
import argparse
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path

import pynvml
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32
from sglang.kernels.ops.diffusion.activation.sana_conv_post_triton import fused_bias_glu


@triton.jit
def probe_kernel(
    X, W, Bias, Raw, Out, N: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, WIDTH: tl.constexpr, ORDER: tl.constexpr,
    SAVE_RAW: tl.constexpr, BLOCK: tl.constexpr,
):
    # X/Out use channels-last storage, weights retain [2C,1,3,3] storage.
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * H * WIDTH * C
    channel = offsets % C
    pixel = offsets // C
    x = pixel % WIDTH
    y = (pixel // WIDTH) % H
    hidden = tl.full((BLOCK,), 0, tl.float32)
    gate = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(9):
        if ORDER == 0:
            tap = k
        elif ORDER == 1:
            tap = 8 - k
        elif ORDER == 2:
            tap = (k % 3) * 3 + k // 3
        elif ORDER == 3:
            tap = 8 - ((k % 3) * 3 + k // 3)
        elif ORDER == 4:
            tap = (k // 3) * 3 + (2 - k % 3)
        else:
            tap = (2 - k // 3) * 3 + k % 3
        dy = tap // 3 - 1
        dx = tap % 3 - 1
        valid = mask & (y + dy >= 0) & (y + dy < H) & (x + dx >= 0) & (x + dx < WIDTH)
        source = (pixel + dy * WIDTH + dx) * (2 * C) + channel
        a = tl.load(X + source, valid, other=0).to(tl.float32)
        b = tl.load(X + source + C, valid, other=0).to(tl.float32)
        wa = tl.load(W + channel * 9 + tap, mask, other=0).to(tl.float32)
        wb = tl.load(W + (channel + C) * 9 + tap, mask, other=0).to(tl.float32)
        hidden = tl.fma(a, wa, hidden)
        gate = tl.fma(b, wb, gate)
    hidden = round_bf16_to_fp32(hidden)
    gate = round_bf16_to_fp32(gate)
    if SAVE_RAW:
        tl.store(Raw + pixel * (2 * C) + channel, hidden, mask)
        tl.store(Raw + pixel * (2 * C) + channel + C, gate, mask)
    bh = tl.load(Bias + channel, mask, other=0).to(tl.float32)
    bg = tl.load(Bias + channel + C, mask, other=0).to(tl.float32)
    hidden = round_bf16_to_fp32(hidden + bh)
    gate = round_bf16_to_fp32(gate + bg)
    gate = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    tl.store(Out + offsets, hidden * gate, mask)


def compare(actual, reference):
    different = actual != reference
    count = int(different.sum().item())
    result = dict(exact=count == 0, different=count, numel=actual.numel())
    if count:
        # Bound diagnostic indexing even if an experimental order is entirely wrong.
        indices = different.flatten()[:65536].nonzero().flatten()[:8].tolist()
        actual_flat, reference_flat = actual.flatten(), reference.flatten()
        result['examples_within_first_65536_logical_elements'] = [
            dict(flat_index=index, actual=float(actual_flat[index]),
                 reference=float(reference_flat[index])) for index in indices]
        result['max_abs_error'] = float((actual.float() - reference.float()).abs().max())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/task/repos/sglang'))
    args = parser.parse_args()
    assert subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'], text=True).strip() == 'af21e2b989e482cfdce39689126562f2209d53ac'
    assert not subprocess.check_output(['git','-C',str(args.repo),'diff','HEAD','--binary','--full-index'])
    assert Path(inspect.getfile(fused_bias_glu)).resolve().is_relative_to((args.repo/'python').resolve())
    pynvml.nvmlInit()
    assert pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(0)) == 'GPU-1565b45e-a702-2015-9776-61f68c36fd59'
    assert all(not pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())), 'GPU already in use; do not overlap timing'
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    result = dict(started_unix=time.time(), torch=torch.__version__, triton=triton.__version__,
                  cudnn=torch.backends.cudnn.version(), source_sha='af21e2b989e482cfdce39689126562f2209d53ac',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  probe_only=True, real_model_quality_or_performance_admitted=False,
                  order_names=['row_major','reverse_row_major','column_major','reverse_column_major',
                               'row_major_reversed_columns','row_major_reversed_rows'], rows=[])
    def save():
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        for seed, shape in [(101, (2, 192, 7, 11)), (202, (21, 13440, 30, 52)),
                            (303, (21, 13440, 30, 52))]:
            n, double_c, h, w = shape
            c = double_c // 2
            generator = torch.Generator(device='cuda:0').manual_seed(seed)
            x = torch.randn((n,h,w,double_c), generator=generator, device='cuda:0', dtype=torch.bfloat16).permute(0,3,1,2)
            weight = torch.randn((double_c,1,3,3), generator=generator, device='cuda:0', dtype=torch.bfloat16) * .1
            bias = torch.randn((double_c,), generator=generator, device='cuda:0', dtype=torch.bfloat16) * .1
            reference_raw = F.conv2d(x, weight, None, padding=1, groups=double_c)
            reference_out = fused_bias_glu(reference_raw, bias)
            raw = torch.empty_like(reference_raw, memory_format=torch.channels_last)
            out = torch.empty_like(reference_out, memory_format=torch.channels_last)
            for order in range(6):
                compiled = probe_kernel[(triton.cdiv(out.numel(),256),)](
                    x,weight,bias,raw,out,n,c,h,w,order,True,256,
                    num_warps=4,enable_fp_fusion=False)
                row = dict(seed=seed,shape=list(shape),input_stride=list(x.stride()),order=order,
                           raw=compare(raw,reference_raw),post=compare(out,reference_out))
                (args.out/f'seed{seed}-order{order}.ptx').write_text(compiled.asm['ptx'])
                result['rows'].append(row)
                save()
                print(json.dumps(row), flush=True)
            del x,weight,bias,reference_raw,reference_out,raw,out
        result['orders_exact_on_all_random_cases'] = [order for order in range(6) if all(
            row['raw']['exact'] and row['post']['exact'] for row in result['rows'] if row['order']==order)]
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
