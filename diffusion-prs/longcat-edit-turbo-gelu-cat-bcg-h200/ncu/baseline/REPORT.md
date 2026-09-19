# LongCat GELU and concatenation baseline

The native trace contains 20 single-stream sites with MLP input `[1, 9233, 12288]` and attention output `[1, 9233, 3072]`, both BF16. The other 20 GELU calls belong to joint-block FFNs and are outside this fusion.

The isolated harness runs the exact PyTorch `F.gelu(approximate="tanh")` followed by `torch.cat`. Environment: H200, Torch 2.13.0+cu130, Torch commit `cf30153c4c131c8164ee7798e5022d810682e2cb`. NCU collected full reports with PM sampling (45 passes per kernel) and source counters (5 passes per kernel). The installed Torch binary has no embedded source lines, so no per-line attribution is claimed for aten.

| Kernel | NCU time | DRAM read | DRAM write | SM throughput | Active warps | Registers |
|---|---:|---:|---:|---:|---:|---:|
| GELU | 159.808 us | 226.945 MB | 207.589 MB | 78.17% | 87.01% | 32 |
| Concat | 142.528 us | 283.650 MB | 263.441 MB | 20.38% | 93.06% | 23 |

The pair takes 302.336 us and transfers 981.626 MB. Concat reaches 79.79% combined read/write peak throughput. GELU reaches 56.53% DRAM throughput and 78.17% SM throughput. Removing the intermediate GELU store and reload is therefore a concrete way to reduce this chain's memory traffic.

These NCU times use profiling clocks. Native saved-request ABBA and the rotating-input marker benchmark provide performance claims; they are not mixed with NCU times.
