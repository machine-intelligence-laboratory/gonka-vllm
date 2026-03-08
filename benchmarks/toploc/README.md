# TOPLOC Data Collection

Collects top-k hidden states and top-k logprobs per generated token for Llama 3.1-8B-Instruct (FP8 and INT4) across two hardware setups. Data is used offline for TOPLOC hash calibration.

## Prerequisites

- Two servers running the instrumented vLLM with `toploc_k` / `toploc_logprobs_k` support
- Both Llama 3.1-8B model weights downloaded on each server:
  - FP8: `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`
  - INT4: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- Python packages: `pydantic`, `requests`, `transformers`, `datasets`, `tqdm`

## Before running

Edit `inference.py` and fill in the server addresses (lines 50-62):

```python
hardware_A = ServerConfig(
    ip='...',
    inference_port='...',
    node_port='...',
    gpu='1xH100',
)

hardware_B = ServerConfig(
    ip='...',
    inference_port='...',
    node_port='...',
    gpu='1xA6000',
)
```

## Run

```bash
cd benchmarks/toploc
python inference.py
```

The script runs 4 experiments sequentially (2x2 matrix: FP8/INT4 x Hardware A/B). Each experiment sends 1000 multilingual prompts (200 per language: en, es, zh, hi, ar), deploys models on both servers, and collects inference + validation results with `toploc_k=512`.

## Results

Output goes to `benchmarks/toploc/data/toploc_results/`. For each run you get two files:

| File | Contents |
|------|----------|
| `<model>_<precision>_<gpu>___<model>_<precision>_<gpu>.jsonl` | One JSON line per prompt with full inference/validation results including toploc data |
| `..._config.json` | Run configuration (model presets, server info, request params) |

Example filenames:
```
Meta-Llama-3.1-8B-Instruct-FP8_fp8_1xH100___Meta-Llama-3.1-8B-Instruct-FP8_fp8_1xA6000.jsonl
Meta-Llama-3.1-8B-Instruct-FP8_fp8_1xH100___Meta-Llama-3.1-8B-Instruct-FP8_fp8_1xA6000_config.json
```

### Loading results

```python
from validation.data import load_from_jsonl

items = load_from_jsonl("data/toploc_results/<filename>.jsonl")
# Each item has:
#   item.inference_result.results[i].toploc_indices   — top-512 hidden state indices
#   item.inference_result.results[i].toploc_values    — top-512 hidden state values
#   item.inference_result.results[i].toploc_logprob_token_ids  — top-512 logprob token IDs
#   item.inference_result.results[i].toploc_logprob_values     — top-512 logprob values
```
