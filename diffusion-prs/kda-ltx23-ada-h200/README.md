# LTX-2.3 H200 Ada-value evidence

Baseline c842d32b0ab45753178c5f6420662d7812766aca; integrated candidate
bc556521a0704b2d21a31dc9d0031093d9b6c5d5. Candidate is Kimi K3's KDA
submission 970fabc2be8f4e22. Both production function ASTs exactly match the
immutable solution in kda-candidate/kernel.py; integration only removes task
wrappers/module text and retains the original license and attribution.

Native model Lightricks/LTX-2.3 at revision
7caa482d5cd10a2eae6b34cb48f093ebc45a263e, prepared with SGLang's existing
native overlay/materializer. model-preparation.json pins the overlay and donor
revisions and verifies shared metadata was unchanged. cross_attention_adaln is
true in the actual checkpoint; ordinary LTX-2's six-row path does not exercise
this nine-output kernel.

Two A-B-B-A groups: one H200, 768x512, 121 frames, 30 steps, seed 42, CFG 3,
24 fps, prompt "A beautiful sunset over the ocean", lossless, eager, resident
components and request warmup. Only the Ada-value source file changes between
arms. All eight MP4 files are identical. Every one of the 121 video frames and
481280 decoded FP32 audio samples agrees exactly. Full per-run audits are in
model-evidence.json; original generated videos and rendered previews are in
media/. Source and runtime version manifests remain stable.

## Explicit audio reproduction condition

Default 49-frame/4-step diagnostic A/B videos were pixel-exact, while audio
max error was 0.010592. A same-source baseline repeat also differed only in audio
(max error 0.010734). These default-condition diagnostics failed the exact audio
gate and are preserved in reproducibility/; they are not accepted ABBA evidence.

The full accepted runs apply the same validation-only context around audio VAE
decode and vocoder forward in both arms: deterministic cuDNN, benchmark off,
cuDNN and matmul TF32 off, with all flags restored afterward. Source is in
scripts/repro/sitecustomize.py and per-run records/hashes show the context ran.
Production model code, checkpoint weights and quality settings are unchanged.
The stated native timing results apply to this explicitly recorded condition.

An additional actual-input shadow diagnostic compares baseline and candidate
within the same native execution: 1728 calls and 15552 outputs pass every-byte,
shape/stride/dtype/offset, alias and input-preservation checks. It is diagnostic
correctness evidence, separate from all native timing.

## Performance scope

Three independent eight-case CUPTI suite repeats give 2.4422x/2.4504x/2.4532x
unweighted geometric speedup. Some suite rows are larger source-contract stress
cases, not observed model shapes. Kernel measurements and native latency are
reported separately; see kernel-benchmark/README.md for reproduction.

Four actual model layouts (49-frame diagnostic, including request warmup)
were independently remeasured with three CUDA-graph repetitions: unweighted
geomean 1.6551x-1.6585x. The saved-generation single-row D4096 layout is
2.6965x-2.7209x and D2048 is 1.6373x-1.6493x. The large request-warmup
layout is 1.0363x-1.0387x. These are captured metadata with regenerated
values, with two input states and exact layout/alias/input checks; see
native-kernel-comparison/. They are separate from the full 121-frame media
validation and the actual-input shadow gate.

Saved-request means: 43.665->43.645s and 43.755->43.780s, effectively unchanged.
No model-level speedup is claimed. Timing excludes diagnostic tracing and uses
fresh model processes, request warmup and one exclusive GPU lease.

The original run commands, environments, logs and source manifests are in
runs/. Scripts preserve campaign paths; adapt paths/ports to reproduce with the
same public checkpoints. Package metadata may name the installed SGLang version;
request.json and source-files.json identify the actual pinned imported source.
The KDA runner uses an explicit host-process adaptation in the existing SGLang
environment and does not claim upstream Docker isolation.
