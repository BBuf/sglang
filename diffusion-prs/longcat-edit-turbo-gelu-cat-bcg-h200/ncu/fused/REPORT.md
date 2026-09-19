# LongCat fused GELU and concatenation

The measured kernel is committed in `a7d440f640`, with identical kernel source in combined candidate `10410a20b6`. Full and source NCU reports completed in retry 1. The first harness failed before profiling because its filename `profile.py` shadowed the Python standard library; it was renamed `run_kernel.py`.

The fused kernel takes 183.360 us under NCU, compared with 302.336 us for the original pair. DRAM traffic falls from 981.626 MB to 548.146 MB, a 44.16% reduction. It reaches 74.63% SM throughput, 62.14% DRAM throughput and 73.03% L2 throughput. It uses 32 registers, no spills, and has 72.23% active warps. Its persistent grid has 1,056 CTAs, based on the queried 132 SMs and occupancy of 8 blocks per SM.

Source sampling identifies the MLP load dependency at `gelu_tanh_cat.cuh:38` as the largest stall site: 5,966 long-scoreboard samples. GELU arithmetic at line 45 has 1,003 wait samples, 400 math-pipe-throttle samples and 579 short-scoreboard samples. Runtime row division at line 29 contributes 299 wait and 184 math-pipe-throttle samples. Both memory dependencies and arithmetic remain relevant.

The rotating-input marker benchmark measures 147.547 us fused versus 266.897 us eager. The native trace replaces 20 GELU calls plus 20 channel concatenations with 20 fused calls taking 3.224354 ms. Total kernels in the complete denoising step fall from 699 to 679. All finite BF16 inputs and the native saved PNGs remain exact.

The next isolated experiment replaced libdevice tanh with PTX `tanh.approx`. It changed 60 finite BF16 outputs for only about 3% additional microbenchmark gain. It was rejected and is not committed. NCU's generic FMA advice is also insufficient justification to change the arithmetic's rounding contract.

Full/source reports, source stall counts and PM sampling sections are retained. No detailed PM time-series conclusion is claimed. Kernel-only ABBA did not meet the E2E acceptance threshold; proper warmup geometry and BCG are being evaluated separately in the combined candidate.
