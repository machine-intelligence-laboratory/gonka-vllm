# TOPLOC Data Collection

Collects top-k hidden states and top-k logprobs per generated token from a running vLLM server. Data is used offline for TOPLOC hash calibration.

## Prerequisites

- vLLM built with `toploc_k` / `toploc_logprobs_k` support
- Model weights downloaded
- Python packages: `pydantic`, `requests`, `transformers`, `datasets`, `tqdm`

## Workflow

### 1. Prepare prompts (once)

Generate a fixed prompt set that will be reused across all runs:

```bash
cd benchmarks/toploc
python prepare_prompts.py --output prompts.json
python prepare_prompts.py --output prompts.json --n-prompts 500 --langs en sp ch
```

### 2. Start vLLM

Launch vLLM with the model you want to collect data from:

```bash
vllm serve RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --dtype float16 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192
```

### 3. Collect

Run the collection script against the running server:

```bash
python collect.py \
    --server-url http://localhost:8000 \
    --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --prompts prompts.json \
    --output-dir vllm_1k_prompt \
    --gpu 1xH100 \
    --precision fp8
```

### 4. Repeat for other models/machines

Stop vLLM, start with a different model or on a different machine, and run `collect.py` again with the **same `prompts.json`** and a new `--output-dir`:

```bash
# Different model
python collect.py \
    --server-url http://localhost:8000 \
    --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --prompts prompts.json \
    --output-dir vllm_1k_prompt \
    --gpu 1xH100 \
    --precision int4

# Different machine — copy prompts.json there, run the same way
python collect.py \
    --server-url http://localhost:8000 \
    --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --prompts prompts.json \
    --output-dir vllm_1k_prompt \
    --gpu 1xA6000 \
    --precision fp8
```

## collect.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--server-url` | (required) | vLLM server URL |
| `--model` | (required) | Model name as served by vLLM |
| `--prompts` | (required) | Path to prompts JSON from `prepare_prompts.py` |
| `--output-dir` | (required) | Directory to save results |
| `--gpu` | `unknown` | GPU label for filenames |
| `--precision` | `unknown` | Precision label for filenames |
| `--max-tokens` | `3000` | Max tokens per completion |
| `--temperature` | `0.99` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--toploc-k` | `512` | Top-k hidden state indices to collect |
| `--toploc-logprobs-k` | `512` | Top-k logprob token IDs to collect |
| `--top-logprobs` | `5` | Number of top logprobs from OpenAI API |
| `--max-workers` | `None` | Thread pool size for concurrent requests |

## Output

Each run produces two files in `--output-dir`:

| File | Contents |
|------|----------|
| `<model>_<precision>_<gpu>.jsonl` | One JSON line per prompt with toploc data |
| `<model>_<precision>_<gpu>_config.json` | Run configuration |

### 5. Evaluate

Run `evaluate.py` to verify toploc proofs across collected data. Supports self-verification (baseline), cross-model, and cross-GPU comparison. Works with both full dense NPZ files and reduced sparse format (top-128 indices + values).

#### Self-verify (baseline — should be 100% exact)

```bash
python evaluate.py \
    --npz-dir vllm_1k_prompt/qwen_int8_a100_5 \
    --mode both --k 64,128 --num-positions 1,2,4,8
```

#### Cross-verify: Int8 vs full precision (same GPU)

```bash
python evaluate.py \
    --npz-dir vllm_1k_prompt/qwen_int8_a100_5 \
    --npz-dir-b vllm_1k_prompt/qwen_full_a100_5 \
    --jsonl vllm_1k_prompt/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_5_free.jsonl \
    --jsonl-b vllm_1k_prompt/Qwen2.5-3B-Instruct_fp8_A100_5_enforced.jsonl \
    --mode both --k 128 --num-positions 1 \
    --output results_int8_vs_full.json
```

#### Cross-verify: Int8 vs Int4 (same GPU)

```bash
python evaluate.py \
    --npz-dir vllm_1k_prompt/qwen_int8_a100_5 \
    --npz-dir-b vllm_1k_prompt/qwen_int4_a100_5 \
    --jsonl vllm_1k_prompt/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_5_free.jsonl \
    --jsonl-b vllm_1k_prompt/Qwen2.5-3B-Instruct-GPTQ-Int4_fp8_A100_5_enforced.jsonl \
    --mode both --k 64,128 --num-positions 1,2,4,8 \
    --output results_int8_vs_int4.json
```

#### Cross-verify: same model, different GPU

```bash
python evaluate.py \
    --npz-dir vllm_1k_prompt/qwen_int8_a100_5 \
    --npz-dir-b vllm_1k_prompt/qwen_int8_a100_7 \
    --jsonl vllm_1k_prompt/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_5_free.jsonl \
    --jsonl-b vllm_1k_prompt/Qwen2.5-3B-Instruct-GPTQ-Int8_fp8_A100_7_enforced.jsonl \
    --mode both --k 64,128 --num-positions 1,2,4,8 \
    --output results_int8_gpu5_vs_gpu7.json
```

#### Results (1k samples, Qwen2.5-3B-Instruct, A100)

| Comparison | Mode | k | exact% | exp_mismatch | mant_err |
|---|---|---|---|---|---|
| self (int8) | hidden_states | 128 | 100% | 0 | 0 |
| self (int8) | logprobs | 128 | 100% | 0 | 0 |
| int8 GPU5 vs GPU7 | hidden_states | 128 | 100% | 0 | 0 |
| int8 GPU5 vs GPU7 | logprobs | 128 | 100% | 0 | 0 |
| int8 vs full | hidden_states | 64 | 0% | 2.08 | 1.91 |
| int8 vs full | hidden_states | 128 | 0% | 3.87 | 1.96 |
| int8 vs full | logprobs | 64 | 0% | 11.19 | 17.90 |
| int8 vs full | logprobs | 128 | 0% | 22.82 | 18.10 |
| int8 vs int4 | hidden_states | 64 | 0% | 14.27 | 11.21 |
| int8 vs int4 | hidden_states | 128 | 0% | 26.97 | 12.45 |
| int8 vs int4 | logprobs | 64 | 0% | 48.00 | ~10^17 |
| int8 vs int4 | logprobs | 128 | 0% | 97.01 | ~10^17 |

Same model on different GPUs produces identical proofs. Quantization level (int8 vs full, int8 vs int4) creates detectable divergence, with int4 logprobs diverging catastrophically.

### Loading results

```python
from validation.data import load_collection_from_jsonl

items = load_collection_from_jsonl("data/experiment1/Meta-Llama-3.1-8B-Instruct-FP8_fp8_1xH100.jsonl")
# Each item has:
#   item.result.results[i].toploc_indices          — top-k hidden state indices
#   item.result.results[i].toploc_values           — top-k hidden state values
#   item.result.results[i].toploc_logprob_token_ids — top-k logprob token IDs
#   item.result.results[i].toploc_logprob_values   — top-k logprob values
```
