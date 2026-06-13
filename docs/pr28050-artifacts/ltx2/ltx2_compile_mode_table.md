| Mode | Runs | Denoise mean | Denoise median | E2E mean | E2E median | Peak mem mean |
|---|---:|---:|---:|---:|---:|---:|
| `max-autotune-no-cudagraphs` | 3 | 131.917 s | 131.971 s | 133.388 s | 133.605 s | 62.3 GB |
| `default` | 3 | 120.410 s | 120.526 s | 121.781 s | 121.894 s | 62.2 GB |

Mean speedup default vs max: denoise 8.72%, E2E 8.70%.
Median speedup default vs max: denoise 8.67%, E2E 8.77%.

Current PR head no-env default confirmation: commit `883c62860b9a94f57e63d2157698137bd8865b31`, no `SGLANG_TORCH_COMPILE_MODE` override, log shows `Compiling transformer with mode: default`; E2E 123.901 s, denoise 121.785 s, peak reserved 62.3 GB.
