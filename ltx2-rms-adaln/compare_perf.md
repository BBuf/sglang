### Performance Comparison Report

#### 1. High-level Summary
| Metric | Baseline | New | Diff | Status |
| :--- | :--- | :--- | :--- | :--- |
| **E2E Latency** | 47389.96 ms | 46261.16 ms | **-1128.80 ms (-2.4%)** | ✅ |
| **Throughput** | 0.02 req/s | 0.02 req/s | - | - |


#### 2. Stage Breakdown
| Stage Name | Baseline (ms) | New (ms) | Diff (ms) | Diff (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| InputValidationStage | 0.06 | 0.03 | -0.02 | -38.0% | ⚪️ |
| TextEncodingStage | 391.39 | 392.51 | +1.12 | +0.3% | ⚪️ |
| LTX2TextConnectorStage | 24.44 | 19.39 | -5.06 | -20.7% | ⚪️ |
| LTX2HalveResolutionStage | 0.07 | 0.04 | -0.03 | -46.0% | ⚪️ |
| LTX2LoRASwitchStage | 3.99 | 2.45 | -1.54 | -38.6% | ⚪️ |
| LTX2SigmaPreparationStage | 0.35 | 0.21 | -0.13 | -38.0% | ⚪️ |
| TimestepPreparationStage | 0.33 | 0.20 | -0.13 | -39.4% | ⚪️ |
| LTX2AVLatentPreparationStage | 0.18 | 0.09 | -0.09 | -48.8% | ⚪️ |
| LTX2ImageEncodingStage | 0.02 | 0.02 | +0.00 | +16.1% | ⚪️ |
| LTX2AVDenoisingStage | 29980.06 | 29442.09 | -537.96 | -1.8% | ⚪️ |
| LTX2UpsampleStage | 4.05 | 3.47 | -0.58 | -14.2% | ⚪️ |
| ltx2_lora_switch_stage2 | 3.85 | 3.40 | -0.44 | -11.5% | ⚪️ |
| ltx2_image_encoding_stage2 | 0.02 | 0.02 | +0.00 | +0.2% | ⚪️ |
| LTX2RefinementStage | 12770.02 | 12663.09 | -106.93 | -0.8% | ⚪️ |
| LTX2AVDecodingStage | 4186.96 | 3711.15 | -475.81 | -11.4% | 🟢 |
| Scheduler.return_result.spill_arrays | 0.13 | 0.15 | +0.02 | +11.7% | ⚪️ |
| SchedulerClient.materialize_file_refs | 0.01 | 0.01 | +0.00 | +7.7% | ⚪️ |


<details>
<summary>Metadata</summary>

- Baseline Commit: `a10a24e9a7fdc0ddcc7dc3e90c0d4983707db028`
- New Commit: `8e8c1ccc065f520f2f9eff69cfa287dd279e59b4`
- Timestamp: 2026-06-26T07:08:15.090917
</details>
