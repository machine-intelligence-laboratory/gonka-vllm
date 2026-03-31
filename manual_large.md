# TOPLOC Collection — Large Models (Qwen3-235B-A22B)

## Server setup

### Install CUDA 12.8

B200 (Blackwell) requires CUDA 12.8+. H100 (Hopper) also works with 12.8. Driver 570.86.15+ required.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8
```

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvidia-smi
nvcc --version
```

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Install vLLM

```bash
cd ~/vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
grep -v '^torch==' requirements/build.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

## Download models

```bash
huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --local-dir /data/models/Qwen3-235B-A22B-Instruct-2507-FP8

huggingface-cli download chriswritescode/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16 \
  --local-dir /data/models/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16
```

## Run 1: INT4 free generation (4xH100, TP=4)

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_int4_4xh100 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16 \
    --tensor-parallel-size 4 \
    --port 8801
```

```bash
cd benchmarks/toploc
python collect.py \
  --gpu 4xH100 \
  --precision int4 \
  --server-url http://localhost:8801 \
  --prompts prompts_all.json \
  --max-tokens 32768 \
  --output-dir ~/vllm_logs
```

## Run 2: FP8 free generation (4xA100, TP=4) — honest reference

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_fp8_4xa100 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --port 8801
```

```bash
python collect.py \
  --gpu 4xA100 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --prompts prompts_all.json \
  --max-tokens 32768 \
  --output-dir ~/vllm_logs
```

## Run 3: FP8 enforced with INT4 tokens (4xH100, TP=4) — negative control

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_fp8_4xh100_enf_from_int4 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --port 8801
```

```bash
python collect.py \
  --gpu 4xH100 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --from-collection ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16_int4_4xH100_free.jsonl \
  --output-dir ~/vllm_logs
```

## Run 4: FP8 enforced with FP8 tokens (4xH100, TP=4) — positive control

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_TOPLOC_OUTPUT_DIR=~/vllm_logs/qwen235b_fp8_4xh100_enf_from_fp8 \
  vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --port 8801
```

```bash
python collect.py \
  --gpu 4xH100 \
  --precision fp8 \
  --server-url http://localhost:8801 \
  --from-collection ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xA100_free.jsonl \
  --output-dir ~/vllm_logs
```

## Evaluate

Negative control (Run 1 vs Run 3 — should fail):
```bash
python evaluate.py \
  --npz-dir ~/vllm_logs/qwen235b_int4_4xh100 \
  --npz-dir-b ~/vllm_logs/qwen235b_fp8_4xh100_enf_from_int4 \
  --jsonl ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16_int4_4xH100_free.jsonl \
  --jsonl-b ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xH100_enforced.jsonl
```

Positive control (Run 2 vs Run 4 — should pass):
```bash
python evaluate.py \
  --npz-dir ~/vllm_logs/qwen235b_fp8_4xa100 \
  --npz-dir-b ~/vllm_logs/qwen235b_fp8_4xh100_enf_from_fp8 \
  --jsonl ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xA100_free.jsonl \
  --jsonl-b ~/vllm_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xH100_enforced.jsonl
```

## Archive and download

```bash
tar czf ~/vllm_archive_large.tar.gz \
  ~/vllm_logs
```

```bash
scp -J <JUMP_USER>@<JUMP_HOST> \
  <USER>@<SERVER_IP>:~/vllm_archive_large.tar.gz .
```
