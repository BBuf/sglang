# Kernel reproduction

These are the frozen KDA harness, definition and six initial workload records
used for the independent integrated-source results. They are source/test-derived
cases, not all captured model layouts. The separate native-layout files record
actual 17-frame and 65-frame model inputs.

Use two SGLang checkouts: baseline c842d32b0ab45753178c5f6420662d7812766aca and
candidate b1c6e3aaa0b7b13ba0d2597d998e066d68e516e8. The runtime versions are in
../runs/hunyuan-kda-r1-a1/environment.json. CUPTI 13 and FlashInfer are required.
Run with one exclusively available H200. The baseline source hash is checked.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/baseline/python
export CANDIDATE_SOURCE=/path/to/candidate/python/sglang/kernels/ops/diffusion/rope/hunyuan_qkv_pack_triton.py
export BENCH_INCLUDE_LARGE=1 BENCH_FAIL_ON_ERROR=1
export BENCH_OUTPUT_JSONL=/tmp/hunyuan-kernel-result.jsonl
python benchmarks/captured_sglang.py --definition diffusion_h200__hunyuan_qkv_rope_pack --candidate integrated_candidate.py
```

Run from this directory in three fresh processes for repeated measurements.
The native full generation and media evidence is separate from kernel timing.
