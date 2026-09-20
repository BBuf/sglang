# LingBot Hopper-only FP32 LayerNorm dispatch

Head:0267caed8ab9872f401b65d586c73fa559d514fc. B200 job106123690318 exposed a changed-input graph-replay mismatch after an initially bit-exact input. Restrict the model FP32 LayerNorm fusion to Hopper; one successful input does not prove another architecture has an equivalent reduction tree. Other GPUs retain native LayerNorm. Residual/camera elementwise paths are unchanged.

H200:9 tests and15 subtests passed, including mandatory Hopper fusion, exact changed-input/parameter replay, native non-Hopper dispatch replay, mismatch fallback and complete block tests. Changed-file pre-commit and Mint validate/broken-links passed. New B200 CI remains required.

Original native E2E/output/NCU measurements remain pinned to89b3700dbc9146727cbb368b6859f5a18e79ae66. No new native model timings are claimed; Hopper numerical kernels are unchanged.
