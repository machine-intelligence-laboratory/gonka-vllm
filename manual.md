# TOPLOC Collection — Small Models (Qwen2.5-3B)

## Install

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
grep -v '^torch==' requirements/build.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

## Download models

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /data/shared/gonka/models/Qwen2.5-3B-Instruct

huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 \
  --local-dir /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int8

huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --local-dir /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int4
```

## Run 1: INT4 free generation (V100)

```bash
env CUDA_VISIBLE_DEVICES=3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen_int4_v100_3 \
  vllm serve /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int4 --port 8801
```

```bash
cd benchmarks/toploc
python collect.py \
  --gpu v100_3 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --prompts prompts.json \
  --output-dir ~/vllm_logs
```

## Run 2: INT8 free generation (A100)

```bash
env CUDA_VISIBLE_DEVICES=5 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen_int8_a100_5 \
  vllm serve /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int8 --port 8801
```

```bash
python collect.py \
  --gpu A100_5 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --prompts prompts.json \
  --output-dir ~/vllm_logs
```

## Run 3: INT8 enforced with INT4 tokens (A100) — negative control

```bash
env CUDA_VISIBLE_DEVICES=7 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen_int8_a100_7_enf_from_int4 \
  vllm serve /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int8 --port 8801
```

```bash
python collect.py \
  --gpu A100_7 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --from-collection ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int4_fp8_v100_3_free.jsonl \
  --output-dir ~/vllm_logs
```

## Run 4: INT8 enforced with INT8 tokens (A100) — positive control

```bash
env CUDA_VISIBLE_DEVICES=7 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen_int8_a100_7_enf_from_int8 \
  vllm serve /data/shared/gonka/models/Qwen2.5-3B-Instruct-GPTQ-Int8 --port 8801
```

```bash
python collect.py \
  --gpu A100_7 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --from-collection ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_5_free.jsonl \
  --output-dir ~/vllm_logs
```

## Evaluate

```bash
python evaluate.py \
  --npz-dir ~/vllm_logs/qwen_int4_v100_3 \
  --npz-dir-b ~/vllm_logs/qwen_int8_a100_7_enf_from_int4 \
  --jsonl ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int4_fp8_v100_3_free.jsonl \
  --jsonl-b ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_7_enforced.jsonl
```

```bash
python evaluate.py \
  --npz-dir ~/vllm_logs/qwen_int8_a100_5 \
  --npz-dir-b ~/vllm_logs/qwen_int8_a100_7_enf_from_int8 \
  --jsonl ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_5_free.jsonl \
  --jsonl-b ~/vllm_logs/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_7_enforced.jsonl
```

## Archive and download

```bash
tar czf ~/vllm_archive.tar.gz \
  ~/vllm_logs
```

```bash
scp -J sshjump@93.175.29.159 \
  zenovkin_n@10.55.228.180:~/vllm_archive.tar.gz .
```
