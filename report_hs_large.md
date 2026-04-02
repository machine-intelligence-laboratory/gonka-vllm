# TOPLOC Hidden-State Verification Report — Qwen3-235B-A22B

## Experiment Setup

- **Dataset**: 1100 prompts (multi-language, from `prompts_all.json`), max 32768 tokens, temperature=0.99, seed=42
- **Base model**: Qwen3-235B-A22B-Instruct-2507 (MoE, 235B total / 22B active params)
- **Variants**:
  - `Qwen3-235B-A22B-Instruct-2507-FP8` — native FP8 quantization
  - `Qwen3-235B-A22B-Instruct-2507-INT4-W4A16` — GPTQ 4-bit (weights only)
- **GPUs**: 4×H100 (same node), tensor-parallel TP=4
- **Hidden dimension**: 4096
- **Collected data**: top-512 hidden state indices/values per generated token
- **Serving**: vLLM with FP8 KV cache
- **Samples**: 1100 matched pairs per experiment

### Sequence length statistics

| Experiment | mean | median | min | max | total tokens |
|-----------:|-----:|-------:|----:|----:|-------------:|
| FP8 free generation | 583.1 | 385 | 6 | 9240 | 641,455 |
| INT4 free generation | 578.1 | 374 | 6 | 10,004 | 635,925 |

### Collection protocol

1. **Free generation**: Reference model generates 1100 responses freely
2. **Enforced generation**: Verification model replays the same token sequences (enforced decoding) to collect its hidden states for the same outputs
3. **Evaluation**: TOPLOC proofs are built from the reference activations and verified against the verification model's activations. All generated tokens are passed to toploc (no position subsampling).

### Data cleaning note

The FP8 free collection was initially started from a wrong branch (collecting top-128 with logprobs), then restarted on the correct branch (top-512, no logprobs) without cleaning the output directory. This resulted in 1354 NPZ files (254 old-format + 1100 new-format). Only the 1100 new-format files were used for evaluation.

### Experiments

| # | Reference | Verification | GPU setup | Purpose |
|:-:|-----------|-------------|-----------|---------|
| 1 | FP8 on 4×H100 (free) | FP8 on 4×H100 (enforced) | same node | Positive control: same model, same hardware |
| 2 | INT4 on 4×H100 (free) | FP8 on 4×H100 (enforced from INT4 tokens) | same node | Negative control: different quantization |

**Note**: Both experiments ran on the same H100 node. A stronger positive control (different nodes) is planned for follow-up.

### Proof encoding

Each proof is a polynomial congruence with `k` uint16 coefficients plus a uint16 modulus: **`2 + 2*k` bytes**.
With `batch_size=N`, one proof covers N consecutive tokens.

### Metrics (per sample averages)

- **avg_proofs**: average number of proofs per sample (ceil(sequence_length / batch_size))
- **proof_bytes**: size of a single proof in bytes (`2 + 2*k`)
- **proof_total_bytes**: total proof storage per sample (ceil(avg_proofs) × proof_bytes)
- **bytes/token**: proof storage cost per generated token (proof_total_bytes / avg_sequence_length)
- **exact%**: percentage of proofs with zero exponent mismatches and zero mantissa error
- **exp_mismatch**: mean exponent mismatches per proof
- **mant_err**: mean mantissa error

## FP8 (4×H100) vs FP8 (4×H100) — same model, same node

Reference: `Qwen3-235B-A22B-Instruct-2507-FP8` on 4×H100 (free generation)
Verify against: `Qwen3-235B-A22B-Instruct-2507-FP8` on 4×H100 (enforced tokens)

| k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| 8 | 8 | 73.3 | 18 | 1314 | 2.25 | 100.0% | 0.0000 | 0 |
| 8 | 16 | 36.9 | 18 | 666 | 1.14 | 100.0% | 0.0000 | 0 |
| 8 | 32 | 18.7 | 18 | 342 | 0.59 | 100.0% | 0.0000 | 0 |
| 8 | 64 | 9.6 | 18 | 180 | 0.31 | 100.0% | 0.0000 | 0 |
| 8 | 128 | 5.1 | 18 | 90 | 0.15 | 100.0% | 0.0000 | 0 |
| 8 | 256 | 2.9 | 18 | 54 | 0.09 | 100.0% | 0.0000 | 0 |
| 8 | 512 | 1.8 | 18 | 36 | 0.06 | 100.0% | 0.0000 | 0 |
| 16 | 8 | 73.3 | 34 | 2482 | 4.26 | 100.0% | 0.0000 | 0 |
| 16 | 16 | 36.9 | 34 | 1258 | 2.16 | 100.0% | 0.0000 | 0 |
| 16 | 32 | 18.7 | 34 | 646 | 1.11 | 100.0% | 0.0000 | 0 |
| 16 | 64 | 9.6 | 34 | 340 | 0.58 | 100.0% | 0.0000 | 0 |
| 16 | 128 | 5.1 | 34 | 170 | 0.29 | 100.0% | 0.0000 | 0 |
| 16 | 256 | 2.9 | 34 | 102 | 0.17 | 100.0% | 0.0000 | 0 |
| 16 | 512 | 1.8 | 34 | 68 | 0.12 | 100.0% | 0.0000 | 0 |
| 32 | 8 | 73.3 | 66 | 4818 | 8.26 | 100.0% | 0.0000 | 0 |
| 32 | 16 | 36.9 | 66 | 2442 | 4.19 | 100.0% | 0.0000 | 0 |
| 32 | 32 | 18.7 | 66 | 1254 | 2.15 | 100.0% | 0.0000 | 0 |
| 32 | 64 | 9.6 | 66 | 660 | 1.13 | 100.0% | 0.0000 | 0 |
| 32 | 128 | 5.1 | 66 | 330 | 0.57 | 100.0% | 0.0000 | 0 |
| 32 | 256 | 2.9 | 66 | 198 | 0.34 | 100.0% | 0.0000 | 0 |
| 32 | 512 | 1.8 | 66 | 132 | 0.23 | 100.0% | 0.0000 | 0 |
| 64 | 8 | 73.3 | 130 | 9490 | 16.28 | 100.0% | 0.0000 | 0 |
| 64 | 16 | 36.9 | 130 | 4810 | 8.25 | 100.0% | 0.0000 | 0 |
| 64 | 32 | 18.7 | 130 | 2470 | 4.24 | 100.0% | 0.0000 | 0 |
| 64 | 64 | 9.6 | 130 | 1300 | 2.23 | 100.0% | 0.0000 | 0 |
| 64 | 128 | 5.1 | 130 | 650 | 1.11 | 100.0% | 0.0000 | 0 |
| 64 | 256 | 2.9 | 130 | 390 | 0.67 | 100.0% | 0.0000 | 0 |
| 64 | 512 | 1.8 | 130 | 260 | 0.45 | 100.0% | 0.0000 | 0 |
| 128 | 8 | 73.3 | 258 | 18834 | 32.30 | 100.0% | 0.0000 | 0 |
| 128 | 16 | 36.9 | 258 | 9546 | 16.37 | 100.0% | 0.0000 | 0 |
| 128 | 32 | 18.7 | 258 | 4902 | 8.41 | 100.0% | 0.0000 | 0 |
| 128 | 64 | 9.6 | 258 | 2580 | 4.42 | 100.0% | 0.0000 | 0 |
| 128 | 128 | 5.1 | 258 | 1290 | 2.21 | 100.0% | 0.0000 | 0 |
| 128 | 256 | 2.9 | 258 | 774 | 1.33 | 100.0% | 0.0000 | 0 |
| 128 | 512 | 1.8 | 258 | 516 | 0.88 | 100.0% | 0.0000 | 0 |
| 256 | 8 | 73.3 | 514 | 37522 | 64.35 | 100.0% | 0.0000 | 0 |
| 256 | 16 | 36.9 | 514 | 19018 | 32.62 | 100.0% | 0.0000 | 0 |
| 256 | 32 | 18.7 | 514 | 9766 | 16.75 | 100.0% | 0.0000 | 0 |
| 256 | 64 | 9.6 | 514 | 5140 | 8.81 | 100.0% | 0.0000 | 0 |
| 256 | 128 | 5.1 | 514 | 2570 | 4.41 | 100.0% | 0.0000 | 0 |
| 256 | 256 | 2.9 | 514 | 1542 | 2.64 | 100.0% | 0.0000 | 0 |
| 256 | 512 | 1.8 | 514 | 1028 | 1.76 | 100.0% | 0.0000 | 0 |
| 512 | 8 | 73.3 | 1026 | 74898 | 128.45 | 100.0% | 0.0000 | 0 |
| 512 | 16 | 36.9 | 1026 | 37962 | 65.10 | 100.0% | 0.0000 | 0 |
| 512 | 32 | 18.7 | 1026 | 19494 | 33.43 | 100.0% | 0.0000 | 0 |
| 512 | 64 | 9.6 | 1026 | 10260 | 17.60 | 100.0% | 0.0000 | 0 |
| 512 | 128 | 5.1 | 1026 | 5130 | 8.80 | 100.0% | 0.0000 | 0 |
| 512 | 256 | 2.9 | 1026 | 3078 | 5.28 | 100.0% | 0.0000 | 0 |
| 512 | 512 | 1.8 | 1026 | 2052 | 3.52 | 100.0% | 0.0000 | 0 |

## INT4 (4×H100) vs FP8 (4×H100) — different quantization, same node

Reference: `Qwen3-235B-A22B-Instruct-2507-INT4-W4A16` on 4×H100 (free generation)
Verify against: `Qwen3-235B-A22B-Instruct-2507-FP8` on 4×H100 (enforced tokens from INT4)

| k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| 8 | 8 | 72.7 | 18 | 1314 | 2.27 | 0.0% | 1.5803 | 9.23e+14 |
| 8 | 16 | 36.6 | 18 | 666 | 1.15 | 0.0% | 1.6063 | 9.5231 |
| 8 | 32 | 18.5 | 18 | 342 | 0.59 | 0.0% | 1.7468 | 9.05e+14 |
| 8 | 64 | 9.5 | 18 | 180 | 0.31 | 0.0% | 1.6640 | 1.76e+15 |
| 8 | 128 | 5.1 | 18 | 90 | 0.16 | 0.0% | 1.7423 | 3.32e+15 |
| 8 | 256 | 2.8 | 18 | 54 | 0.09 | 0.0% | 1.9254 | 5.91e+15 |
| 8 | 512 | 1.7 | 18 | 36 | 0.06 | 0.0% | 2.0910 | 9.59e+15 |
| 16 | 8 | 72.7 | 34 | 2482 | 4.29 | 0.0% | 3.1335 | 10.6966 |
| 16 | 16 | 36.6 | 34 | 1258 | 2.18 | 0.0% | 2.9814 | 10.0115 |
| 16 | 32 | 18.5 | 34 | 646 | 1.12 | 0.0% | 3.1230 | 9.3558 |
| 16 | 64 | 9.5 | 34 | 340 | 0.59 | 0.0% | 3.3472 | 8.3777 |
| 16 | 128 | 5.1 | 34 | 170 | 0.29 | 0.0% | 3.1186 | 8.0258 |
| 16 | 256 | 2.8 | 34 | 102 | 0.18 | 0.0% | 3.3477 | 8.0062 |
| 16 | 512 | 1.7 | 34 | 68 | 0.12 | 0.0% | 3.6781 | 8.1608 |
| 32 | 8 | 72.7 | 66 | 4818 | 8.33 | 0.0% | 5.9538 | 11.6289 |
| 32 | 16 | 36.6 | 66 | 2442 | 4.22 | 0.0% | 6.0672 | 10.4758 |
| 32 | 32 | 18.5 | 66 | 1254 | 2.17 | 0.0% | 5.6803 | 9.9274 |
| 32 | 64 | 9.5 | 66 | 660 | 1.14 | 0.0% | 6.1725 | 9.2570 |
| 32 | 128 | 5.1 | 66 | 330 | 0.57 | 0.0% | 6.5545 | 8.3712 |
| 32 | 256 | 2.8 | 66 | 198 | 0.34 | 0.0% | 6.1214 | 8.2751 |
| 32 | 512 | 1.7 | 66 | 132 | 0.23 | 0.0% | 6.6911 | 8.4538 |
| 64 | 8 | 72.7 | 130 | 9490 | 16.42 | 0.0% | 12.2069 | 12.4858 |
| 64 | 16 | 36.6 | 130 | 4810 | 8.32 | 0.0% | 11.7896 | 11.4570 |
| 64 | 32 | 18.5 | 130 | 2470 | 4.27 | 0.0% | 11.8563 | 10.3620 |
| 64 | 64 | 9.5 | 130 | 1300 | 2.25 | 0.0% | 10.9874 | 9.9389 |
| 64 | 128 | 5.1 | 130 | 650 | 1.12 | 0.0% | 12.2313 | 9.3230 |
| 64 | 256 | 2.8 | 130 | 390 | 0.67 | 0.0% | 12.8412 | 8.6256 |
| 64 | 512 | 1.7 | 130 | 260 | 0.45 | 0.0% | 12.3229 | 8.8187 |
| 128 | 8 | 72.7 | 258 | 18834 | 32.58 | 0.0% | 27.6869 | 13.5115 |
| 128 | 16 | 36.6 | 258 | 9546 | 16.51 | 0.0% | 23.5787 | 12.4182 |
| 128 | 32 | 18.5 | 258 | 4902 | 8.48 | 0.0% | 23.5205 | 11.3714 |
| 128 | 64 | 9.5 | 258 | 2580 | 4.46 | 0.0% | 23.3966 | 10.3464 |
| 128 | 128 | 5.1 | 258 | 1290 | 2.23 | 0.0% | 21.7054 | 10.0673 |
| 128 | 256 | 2.8 | 258 | 774 | 1.34 | 0.0% | 24.3484 | 9.6069 |
| 128 | 512 | 1.7 | 258 | 516 | 0.89 | 0.0% | 25.3141 | 9.3017 |
| 256 | 8 | 72.7 | 514 | 37522 | 64.91 | 0.0% | 62.9837 | 14.3115 |
| 256 | 16 | 36.6 | 514 | 19018 | 32.90 | 0.0% | 55.7170 | 13.3986 |
| 256 | 32 | 18.5 | 514 | 9766 | 16.89 | 0.0% | 46.1093 | 12.3980 |
| 256 | 64 | 9.5 | 514 | 5140 | 8.89 | 0.0% | 47.2202 | 11.3640 |
| 256 | 128 | 5.1 | 514 | 2570 | 4.45 | 0.0% | 46.6905 | 10.4641 |
| 256 | 256 | 2.8 | 514 | 1542 | 2.67 | 0.0% | 43.9789 | 10.3279 |
| 256 | 512 | 1.7 | 514 | 1028 | 1.78 | 0.0% | 49.0962 | 10.1611 |
| 512 | 8 | 72.7 | 1026 | 74898 | 129.56 | 0.0% | 146.1542 | 14.6749 |
| 512 | 16 | 36.6 | 1026 | 37962 | 65.67 | 0.0% | 125.7782 | 14.2563 |
| 512 | 32 | 18.5 | 1026 | 19494 | 33.72 | 0.0% | 111.7875 | 13.3419 |
| 512 | 64 | 9.5 | 1026 | 10260 | 17.75 | 0.0% | 90.9816 | 12.4400 |
| 512 | 128 | 5.1 | 1026 | 5130 | 8.87 | 0.0% | 95.2232 | 11.4774 |
| 512 | 256 | 2.8 | 1026 | 3078 | 5.32 | 0.0% | 94.2056 | 10.8005 |
| 512 | 512 | 1.7 | 1026 | 2052 | 3.55 | 0.0% | 90.4306 | 10.9018 |

## Summary

- **Same model, same node (FP8 vs FP8)**: 100% exact match across all 49 configurations (k=8..512, batch_size=8..512). TOPLOC hidden-state proofs are fully reproducible. Zero false negatives.
- **Different quantization, same node (INT4 vs FP8)**: 0% exact match across all 49 configurations. Every single proof correctly detects the quantization mismatch. Zero false positives. Exponent mismatches scale roughly linearly with k (from ~1.6 at k=8 to ~146 at k=512). Mantissa errors at k=8 occasionally overflow to ~1e14–1e15 due to catastrophic exponent divergence; at k≥16 they stabilize around 8–15.
- **No false positives or false negatives**: The hidden-state channel achieves perfect separation between the positive and negative controls at every (k, batch_size) combination tested.
- **Scale-up validation**: These results on a 235B-parameter MoE model (4096 hidden dim, TP=4) confirm the same perfect separation previously observed on the 3B-parameter Qwen2.5 dense model (2048 hidden dim, TP=1).
- **Caveat**: The positive control ran on the same H100 node (different vLLM server instances). A cross-node test is needed to confirm reproducibility across hardware.
