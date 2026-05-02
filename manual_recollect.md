# TOPLOC Re-collect — FP8 free + enforced (logprob values)

## Run 1: FP8 free generation (4xA100, TP=4)

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_fp8_4xa100 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --port 8801 \
    --logprobs-mode raw_logprobs
```

```bash
cd benchmarks/toploc
python collect.py \
  --gpu 4xA100 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --prompts prompts_all.json \
  --max-tokens 32768 \
  --output-dir ~/vllm_logs
```

## Run 2: FP8 enforced validation (4xH100, TP=4)

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_fp8_4xh100_enf_from_fp8 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --port 8801 \
    --logprobs-mode raw_logprobs
```

```bash
python collect.py \
  --gpu 4xH100 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --from-collection ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xA100_free.jsonl \
  --output-dir ~/vllm_logs
```
