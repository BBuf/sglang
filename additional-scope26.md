# Additional scope checks for PR50602, unchanged head37999f7a

Frozen main0745bfe4, Omni aa2968e8, final PR37999f7a. Same physical RTX PRO6000 ServerEdition GPUs, native backend, canonical weights and fixed inputs. This supplement adds correctness/serving evidence; it adds **no performance or BCG claim**. The quiet49-frame ABBA and cleanup results in the original evidence commit remain the only model performance claim.

| Additional scope | Actual main/PR agreement | Native execution |
|---|---|---|
| Single GPU,241frames,short/long,3seeds+repeat |7MP4pairs and1687RGBframes exact;SSIM1,PSNR infinity,max error0 |4332fused mean calls/request |
| Ulysses2,49/121frames,short/long+repeat |6MP4pairs exact same-budget main; each framework also exact its own SP1 outputs |2actual rank receipts;361/1083fused calls per rank for49/121frames |
| Two outstanding API requests,49frames,3pairs |6PR outputs exact main C1/C2; both frameworks'6C2 outputs exact own C1 |Two nonterminal jobs observed after both POSTs;722fused calls/PR request |

The original17main/PR media pairs plus19additional pairs give36byte-identical pairs. No assumption of kernel concurrency or throughput benefit follows from two outstanding API jobs. Shared-code compatibility is proven only for these exercised combinations.

Cross-framework quality remains excluded from performance admission: all6new241-frame pairs fail the unchangedSSIM>=0.92/PSNR>=24dB gate, bringing the SP1 total to18failed pairs; all4SP2 cross pairs fail as well. Six actual241-frame worst panels were inspected: crop/geometry,cables/spokes,text and shadows differ. Both arms producing plausible bicycles does not waive the numeric failures. Native noise/masks/schedules match. No single-cause attribution or Omni speedup is claimed.

Actual first/middle/last main/PR panels are in `evidence/quality/nomas-241-main-cleanup25/`; full decoded comparisons cover all241frames. Their `summary.json` maps to real generated MP4s under the two241-frame run directories. SP2/C2 output hashes link to identical native SP1 content in the original evidence archive; per-request raw replies,rank observations and concurrent status snapshots are included. Full private runtime tensors and media are retained on the task machine and archived separately; no weights are included here.

| Additional cross-framework cell | Minimum all-frame SSIM | Minimum PSNR dB | Frozen gate |
|---|---:|---:|---|
| SP1 nomas-241-short-seed42 | 0.554790 | 16.163 | fail |
| SP1 nomas-241-long-seed42 | 0.588158 | 16.715 | fail |
| SP1 nomas-241-short-seed1101 | 0.585841 | 17.107 | fail |
| SP1 nomas-241-long-seed1101 | 0.729817 | 20.465 | fail |
| SP1 nomas-241-short-seed2026 | 0.661110 | 17.516 | fail |
| SP1 nomas-241-long-seed2026 | 0.635547 | 19.111 | fail |
| SP2 nomas-49-short-seed42 | 0.877107 | 24.427 | fail |
| SP2 nomas-49-long-seed42 | 0.550778 | 16.324 | fail |
| SP2 nomas-121-short-seed42 | 0.647971 | 18.074 | fail |
| SP2 nomas-121-long-seed42 | 0.778515 | 21.883 | fail |
