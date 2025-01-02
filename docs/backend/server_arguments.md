# Server Arguments

- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
```
- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
```
- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
```
- See [hyperparameter tuning](../references/hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
```
- To enable torch.compile acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. This does not work for FP8 currently.
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other [quantization strategies (INT8/FP8)](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L671) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](../references/custom_chat_template.md).

- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 1
```

## Use Models From ModelScope
<details>
<summary>More</summary>

To use a model from [ModelScope](https://www.modelscope.cn), set the environment variable SGLANG_USE_MODELSCOPE.
```
export SGLANG_USE_MODELSCOPE=true
```
Launch [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) Server
```
SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

Or start it by docker.
```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

</details>

## Example: Run Llama 3.1 405B
<details>
<summary>More</summary>

```bash
# Run 405B (fp8) on a single node
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8

# Run 405B (fp16) on two nodes
## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0

## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 1
```

</details>