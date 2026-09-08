# SANA-WM H200 benchmark artifacts

Native model measurement results and output comparisons for the SANA-WM conv/GDN PR.
Each MP4 is the actual saved lossless A1/B1 output; all eight formal outputs for that
model have the same hash, including high quality mode. Contact sheets show baseline
above candidate. The baseline row is labeled `lossless`.

`benchmark-results.json` has all 16 formal measurement rows and recorded commits.
These warm-request numbers exclude loading, warmup, and profiling.

The two `bench_*.py` files are the **original exploratory prototypes**, to reproduce
on measured baseline commit `77fb7a72889a7155b0af34739a409d4517219530` with PyTorch
2.11.0+cu130 on one idle H200. They require no model weights. Set `CUDA_VISIBLE_DEVICES`
to the chosen card. The conv experiment writes to the `SANA_BENCH_RESULT` environment
path; the reverse experiment writes to `SANA_REVERSE_RESULT`. Their JSON results are
included separately. They are device CUDA-graph microbenchmarks, not native model
BCG support measurements; the two native SANA-WM pipelines disable BCG.
