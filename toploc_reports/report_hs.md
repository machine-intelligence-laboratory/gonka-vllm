# TOPLOC Hidden-State Verification Report

## Experiment Setup

- **Dataset**: 1000 prompts (multi-language), max 3000 tokens, temperature=0.99, seed=42
- **Base model family**: Qwen2.5-3B-Instruct
- **Models**:
  - `Qwen2.5-3B-Instruct-GPTQ-Int8` — GPTQ 8-bit quantization
  - `Qwen2.5-3B-Instruct-GPTQ-Int4` — GPTQ 4-bit quantization
- **GPUs**: NVIDIA A100 (two separate cards: A100_5, A100_6)
- **Collected data**: top-128 hidden state indices/values per generated token
- **Serving**: vLLM with FP8 KV cache
- **Samples**: 1000

### Sequence length statistics

| Experiment | mean | median | min | max |
|-----------:|-----:|-------:|----:|----:|
| Int8 free generation | 261.6 | 208 | 5 | 3000 |
| Int4 free generation | 266.6 | 202 | 4 | 3000 |

### Collection protocol

1. **Free generation**: Reference model generates 1000 responses freely
2. **Enforced generation**: Verification model replays the same token sequences (enforced decoding) to collect its hidden states for the same outputs
3. **Evaluation**: TOPLOC proofs are built from the reference activations and verified against the verification model's activations. All generated tokens are passed to toploc (no position subsampling).

### Experiments

| # | Reference | Verification | GPU setup | Purpose |
|:-:|-----------|-------------|-----------|---------|
| 1 | Int8 on A100_5 (free) | Int8 on A100_6 (enforced) | different GPUs | Positive control: same model, different hardware |
| 2 | Int4 on A100_5 (free) | Int8 on A100_5 (enforced from Int4 tokens) | same GPU | Negative control: different quantization |

### Proof encoding

Each proof is a polynomial congruence with `k` uint16 coefficients plus a uint16 modulus: **`2 + 2*k` bytes**.
With `batch_size=N`, one proof covers N consecutive tokens.

### Metrics (per sample averages)

- **avg_proofs**: average number of proofs per sample (ceil(sequence_length / batch_size))
- **proof_bytes**: size of a single proof in bytes (`2 + 2*k`)
- **proof_total_bytes**: total proof storage per sample (ceil(avg_proofs) x proof_bytes)
- **bytes/token**: proof storage cost per generated token (proof_total_bytes / avg_sequence_length)
- **exact%**: percentage of proofs with zero exponent mismatches and zero mantissa error
- **exp_mismatch**: mean exponent mismatches per proof
- **mant_err**: mean mantissa error

## Int8 (A100_5) vs Int8 (A100_6) — same model, different GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_6 (enforced tokens)

| k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| 8 | 8 | 33.1 | 18 | 612 | 2.34 | 100.0% | 0.0000 | 0 |
| 8 | 16 | 16.8 | 18 | 306 | 1.17 | 100.0% | 0.0000 | 0 |
| 8 | 32 | 8.7 | 18 | 162 | 0.62 | 100.0% | 0.0000 | 0 |
| 8 | 64 | 4.6 | 18 | 90 | 0.34 | 100.0% | 0.0000 | 0 |
| 8 | 128 | 2.6 | 18 | 54 | 0.21 | 100.0% | 0.0000 | 0 |
| 8 | 256 | 1.6 | 18 | 36 | 0.14 | 100.0% | 0.0000 | 0 |
| 16 | 8 | 33.1 | 34 | 1156 | 4.42 | 100.0% | 0.0000 | 0 |
| 16 | 16 | 16.8 | 34 | 578 | 2.21 | 100.0% | 0.0000 | 0 |
| 16 | 32 | 8.7 | 34 | 306 | 1.17 | 100.0% | 0.0000 | 0 |
| 16 | 64 | 4.6 | 34 | 170 | 0.65 | 100.0% | 0.0000 | 0 |
| 16 | 128 | 2.6 | 34 | 102 | 0.39 | 100.0% | 0.0000 | 0 |
| 16 | 256 | 1.6 | 34 | 68 | 0.26 | 100.0% | 0.0000 | 0 |
| 32 | 8 | 33.1 | 66 | 2244 | 8.58 | 100.0% | 0.0000 | 0 |
| 32 | 16 | 16.8 | 66 | 1122 | 4.29 | 100.0% | 0.0000 | 0 |
| 32 | 32 | 8.7 | 66 | 594 | 2.27 | 100.0% | 0.0000 | 0 |
| 32 | 64 | 4.6 | 66 | 330 | 1.26 | 100.0% | 0.0000 | 0 |
| 32 | 128 | 2.6 | 66 | 198 | 0.76 | 100.0% | 0.0000 | 0 |
| 32 | 256 | 1.6 | 66 | 132 | 0.50 | 100.0% | 0.0000 | 0 |
| 64 | 8 | 33.1 | 130 | 4420 | 16.90 | 100.0% | 0.0000 | 0 |
| 64 | 16 | 16.8 | 130 | 2210 | 8.45 | 100.0% | 0.0000 | 0 |
| 64 | 32 | 8.7 | 130 | 1170 | 4.47 | 100.0% | 0.0000 | 0 |
| 64 | 64 | 4.6 | 130 | 650 | 2.48 | 100.0% | 0.0000 | 0 |
| 64 | 128 | 2.6 | 130 | 390 | 1.49 | 100.0% | 0.0000 | 0 |
| 64 | 256 | 1.6 | 130 | 260 | 0.99 | 100.0% | 0.0000 | 0 |
| 128 | 8 | 33.1 | 258 | 8772 | 33.53 | 100.0% | 0.0000 | 0 |
| 128 | 16 | 16.8 | 258 | 4386 | 16.77 | 100.0% | 0.0000 | 0 |
| 128 | 32 | 8.7 | 258 | 2322 | 8.88 | 100.0% | 0.0000 | 0 |
| 128 | 64 | 4.6 | 258 | 1290 | 4.93 | 100.0% | 0.0000 | 0 |
| 128 | 128 | 2.6 | 258 | 774 | 2.96 | 100.0% | 0.0000 | 0 |
| 128 | 256 | 1.6 | 258 | 516 | 1.97 | 100.0% | 0.0000 | 0 |

## Int4 (A100_5) vs Int8 (A100_5) — different quantization, same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int4` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (enforced tokens from Int4)

| k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| 8 | 8 | 33.7 | 18 | 612 | 2.30 | 0.0% | 1.4290 | 2.19e+15 |
| 8 | 16 | 17.1 | 18 | 324 | 1.22 | 0.0% | 1.4998 | 5.38e+15 |
| 8 | 32 | 8.8 | 18 | 162 | 0.61 | 0.0% | 1.6984 | 1.68e+16 |
| 8 | 64 | 4.7 | 18 | 90 | 0.34 | 0.0% | 1.8976 | 1.19e+16 |
| 8 | 128 | 2.6 | 18 | 54 | 0.20 | 0.0% | 2.0594 | 1.40e+16 |
| 8 | 256 | 1.6 | 18 | 36 | 0.14 | 0.0% | 2.1287 | 1.13e+16 |
| 16 | 8 | 33.7 | 34 | 1156 | 4.34 | 0.0% | 2.8099 | 2.19e+15 |
| 16 | 16 | 17.1 | 34 | 612 | 2.30 | 0.0% | 2.6942 | 1.08e+15 |
| 16 | 32 | 8.8 | 34 | 306 | 1.15 | 0.0% | 2.8803 | 7.9440 |
| 16 | 64 | 4.7 | 34 | 170 | 0.64 | 0.0% | 3.2541 | 7.4020 |
| 16 | 128 | 2.6 | 34 | 102 | 0.38 | 0.0% | 3.5868 | 6.9506 |
| 16 | 256 | 1.6 | 34 | 68 | 0.26 | 0.0% | 3.8304 | 6.8156 |
| 32 | 8 | 33.7 | 66 | 2244 | 8.42 | 0.0% | 5.1041 | 10.7401 |
| 32 | 16 | 17.1 | 66 | 1188 | 4.46 | 0.0% | 5.5380 | 1.08e+15 |
| 32 | 32 | 8.8 | 66 | 594 | 2.23 | 0.0% | 5.1033 | 8.4968 |
| 32 | 64 | 4.7 | 66 | 330 | 1.24 | 0.0% | 5.5797 | 8.0203 |
| 32 | 128 | 2.6 | 66 | 198 | 0.74 | 0.0% | 6.1929 | 7.6165 |
| 32 | 256 | 1.6 | 66 | 132 | 0.50 | 0.0% | 6.7138 | 7.4092 |
| 64 | 8 | 33.7 | 130 | 4420 | 16.58 | 0.0% | 12.0374 | 11.5080 |
| 64 | 16 | 17.1 | 130 | 2340 | 8.78 | 0.0% | 10.0378 | 10.6686 |
| 64 | 32 | 8.8 | 130 | 1170 | 4.39 | 0.0% | 11.0008 | 9.2841 |
| 64 | 64 | 4.7 | 130 | 650 | 2.44 | 0.0% | 9.8621 | 8.6450 |
| 64 | 128 | 2.6 | 130 | 390 | 1.46 | 0.0% | 10.8577 | 8.2920 |
| 64 | 256 | 1.6 | 130 | 260 | 0.98 | 0.0% | 11.8670 | 8.1065 |
| 128 | 8 | 33.7 | 258 | 8772 | 32.90 | 0.0% | 25.5844 | 13.2078 |
| 128 | 16 | 17.1 | 258 | 4644 | 17.42 | 0.0% | 24.1010 | 11.4206 |
| 128 | 32 | 8.8 | 258 | 2322 | 8.71 | 0.0% | 20.0417 | 10.6632 |
| 128 | 64 | 4.7 | 258 | 1290 | 4.84 | 0.0% | 21.8930 | 9.3813 |
| 128 | 128 | 2.6 | 258 | 774 | 2.90 | 0.0% | 19.6423 | 8.9243 |
| 128 | 256 | 1.6 | 258 | 516 | 1.94 | 0.0% | 21.2923 | 8.7777 |

## Summary

- **Same model, different GPU (Int8 vs Int8)**: 100% exact match across all 30 configurations (k=8..128, batch_size=8..256). TOPLOC hidden-state proofs are fully reproducible across A100 cards. Zero false negatives.
- **Different quantization, same GPU (Int4 vs Int8)**: 0% exact match across all 30 configurations. Every single proof correctly detects the quantization mismatch. Zero false positives. Exponent mismatches scale roughly linearly with k (from ~1.4 at k=8 to ~25 at k=128). Mantissa errors at low k occasionally overflow to ~1e15-1e16 due to catastrophic exponent divergence; at higher k they stabilize around 7-13.
- **No false positives or false negatives**: The hidden-state channel achieves perfect separation between the positive and negative controls at every (k, batch_size) combination tested.