Native FLUX.2 Klein Base4B on one H200: lossless eager packed Q/K RMSNorm + interleaved RoPE.

final-evidence.json retains all initial one-step-warmup groups and the separately fixed full-50-step-warmup groups. Only the latter supply the headline. Profile and NCU are diagnostic. Kernel-only CUDA Graph is not model BCG; all native BCG probes were disabled and excluded. Original baseline/candidate and high PNGs are saved native outputs. The contact grid only resizes them for display.

The v1 kernel failed a signed-zero replay case; v2 fixes sign-bit handling and v3 removes gather by holding rotary pairs in registers. All failed logs remain. The initial quality auditor failed by reading a missing hash field from a deliberately invalid BCG result. Its retry independently hashes the saved fallback PNG; no invalid run becomes a performance result. The consequent premeasurement driver exits are retained.

See scripts, source.diff, profiles, ncu, raw and validation-logs for reproduction. Checkpoint cleanup audited zero remaining owned files/weights/bytes.
