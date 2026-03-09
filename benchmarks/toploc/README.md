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
    --output-dir ./data/experiment1 \
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
    --output-dir ./data/experiment1 \
    --gpu 1xH100 \
    --precision int4

# Different machine — copy prompts.json there, run the same way
python collect.py \
    --server-url http://localhost:8000 \
    --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --prompts prompts.json \
    --output-dir ./data/experiment1 \
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
