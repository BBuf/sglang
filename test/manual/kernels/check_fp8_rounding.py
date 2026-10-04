import importlib.util
from pathlib import Path
import torch

path = Path(__file__).resolve().parents[2] / 'registered/kernels/ops/attention/test_fused_store_index_cache.py'
spec = importlib.util.spec_from_file_location('fused_store_test', path)
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
print('GPU', torch.cuda.get_device_name(), 'torch', torch.__version__, flush=True)


def check(key, tag):
    n = len(key)
    loc = torch.arange(n, device='cuda', dtype=torch.int64)
    pages = (n + 63) // 64
    buf = test._make_buffer(pages)
    test.fused_store_index_k_cache(key, buf, loc)
    out_bytes = buf[:, :64 * 128].reshape(-1, 128)[:n]
    out = out_bytes.view(test.FP8_DTYPE).float()
    row_max = key.float().abs().amax(dim=1).double().clamp(min=1e-4)
    scales = row_max / 448.0
    inv_scale = scales.reciprocal().float()
    ref = (key.float() * inv_scale[:, None]).clamp(-448, 448).to(test.FP8_DTYPE).float()
    # Check vectorized reference against the existing scalar implementation.
    if tag == 'boundary':
        old = test._reference_quantize_and_store(key, loc, pages)
        old_values = old[:, :64*128].reshape(-1, 128)[:n].view(test.FP8_DTYPE).float()
        torch.testing.assert_close(old_values, ref, rtol=0, atol=0)
    a = out.to(test.FP8_DTYPE).view(torch.uint8).to(torch.int16)
    b = ref.to(test.FP8_DTYPE).view(torch.uint8).to(torch.int16)
    a = torch.where(a < 128, a, 128 - a)
    b = torch.where(b < 128, b, 128 - b)
    ulp = (a - b).abs()
    rel = (out - ref).abs() / ref.abs().clamp(min=1e-6)
    bad = (rel > .15).nonzero()
    if len(bad):
        for i, j in bad[:8].tolist():
            print(dict(tag=tag, row=i, col=j, value=key[i,j].item(), absmax=row_max[i].item(), out=out[i,j].item(), ref=ref[i,j].item(), out_code=a[i,j].item(), ref_code=b[i,j].item(), rel=rel[i,j].item(), ulp=ulp[i,j].item()), flush=True)
    assert ulp.max().item() <= 1, (tag, ulp.max().item())
    print(tag, 'elements', n*128, 'mismatches', (out != ref).sum().item(), 'max_ulp', ulp.max().item(), 'old_assert_failures', len(bad), flush=True)
    return len(bad)

key = torch.zeros((4, 128), device='cuda', dtype=torch.bfloat16)
key[:, 0] = torch.tensor([2., 2.0625, 2.21875, 2.328125], device='cuda')
key[:, 1] = key[:, 0] / 65536
key[1::2, 1] *= -1
found = check(key, 'boundary')
for seed in range(20):
    torch.manual_seed(seed)
    key = torch.randn((4096, 128), device='cuda', dtype=torch.bfloat16)
    found += check(key, f'seed={seed}')
print('Total failures of original relative assertion:', found, flush=True)
assert found > 0
