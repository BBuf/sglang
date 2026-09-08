# LongCat LayerNorm/modulation H200 artifacts

Actual saved baseline/PR PNGs for all three checkpoints and both quality modes.
`benchmark-results.json` records all28formal rows including the valid Image lossless
BCG matrix. Baseline/PR files match byte-for-byte within each quality/model.

The included microbenchmark calls the actual existing fused kernel and the original
PyTorch formula. Run it with `PYTHONPATH=python` from a SGLang checkout and select an
idle H200 via `CUDA_VISIBLE_DEVICES`. It requires no model weights. The original
experiment used measured source295e64d7000cfbb4d3bf9f4e46efcdacbc4d84a8.
