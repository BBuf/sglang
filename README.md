# MiniMax-H3-Image iterative edit benchmark

`reproduce.py` reads the SGLang server options and multipart requests from `workload.json`. Use it with SGLang `075ccb8add6d69f5a1954fae1bce3920adc422ef` or `744373f6b54402b0cdd413c0269d96eac031e96b` installed in the active environment. The model is MiniMax-H3-Image at `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`. Server and client run on the same host so the client can read the per-request stage timings written by the server.

```bash
hf download MiniMaxAI/MiniMax-H3-Image --revision 5f90473023975e6271b0ed1dd84c4e88e4bbc6d3 --local-dir h3-image
CUDA_VISIBLE_DEVICES=0 python reproduce.py serve --model-path h3-image
```

Run the client in a second terminal, in the same directory. Restart the server and choose a new output directory for each visit.

```bash
python reproduce.py run --model-path h3-image --out visit-A1
```

The measurements use a single RTX PRO 6000 Blackwell Server Edition with 97,887 MiB. Both server and client were bound to the GPU's NUMA node (`taskset -c 0-31,64-95` on the test host). Dependencies included PyTorch 2.14.1+cu130, Transformers 5.19.0, FlashInfer 0.7.4, Triton 3.8.0, cuDNN 9.24.0.43, Pillow 11.3.0 and Requests 2.34.2.

Each fresh visit submits two warmups, six timed requests with changing text over two familiar references, two new-reference controls and a repeat. `measurements.json` contains the two ABBA rounds and all 88 response latencies and hashes. `pixel-hashes.json` pairs main and head for one complete visit. `main.png` and `head.png` are the full 2048×2048 response to request index 3. `comparison.jpg` shows that pair at reduced resolution. `profile-summary.json` records CPU and GPU attribution from a separate main profiling visit.

The two `ref0` PNGs are 1024×1024 RGB resizes of the example distributed with the pinned MiniMax-H3-Image snapshot. The two `ref1` PNGs are resizes of a separate SGLang main generation on the same snapshot. The `-1` suffix indicates a horizontal flip. [Original MiniMax reference](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/5f90473023975e6271b0ed1dd84c4e88e4bbc6d3/examples/reference1.png)

## Compatibility controls

Add the same `--control MODE` to the server and client commands. Available modes in `control-workloads.json` include `graph-eager`, `graph`, `small-cache`, `layer-offload`, `layer-offload-fp8`, `quality`, `tp2` and `tp2-memory`. Restart the server for each revision and mode. The graph modes alternate prompts over the same reference and set the text bucket to 1088. The quality control checks the model's HTTP 400 response between supported requests. The TP2 modes use `CUDA_VISIBLE_DEVICES=0,1` and select the NCCL path for inter-socket PCIe.

For two independent replicas, start two servers, pin each process to its GPU's CPU NUMA node, and use a different port for each. Give each client the matching port and control. Create a new, shared empty directory for the two clients and pass it as `--barrier-dir` to synchronize their post-warmup requests.

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-31,64-95 python reproduce.py serve --model-path h3-image --control replica0 --port 32150
CUDA_VISIBLE_DEVICES=1 taskset -c 32-63,96-127 python reproduce.py serve --model-path h3-image --control replica1 --port 32152
```

In a separate shell with both servers ready, run the clients together.

```bash
mkdir pair-barrier
taskset -c 0-31,64-95 python reproduce.py run --model-path h3-image --control replica0 --port 32150 --barrier-dir pair-barrier --out replica0 &
taskset -c 32-63,96-127 python reproduce.py run --model-path h3-image --control replica1 --port 32152 --barrier-dir pair-barrier --out replica1 &
wait
```
