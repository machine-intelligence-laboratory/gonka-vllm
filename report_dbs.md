# TOPLOC Cross-Verification Report

## Experiment Setup

- **Dataset**: 1000 prompts (multi-language), max 3000 tokens, temperature=0.99, seed=42
- **Base model family**: Qwen2.5-3B-Instruct
- **Models**:
  - `Qwen2.5-3B-Instruct-GPTQ-Int8` — GPTQ 8-bit quantization
  - `Qwen2.5-3B-Instruct-GPTQ-Int4` — GPTQ 4-bit quantization
  - `Qwen2.5-3B-Instruct` — full precision (BF16)
- **GPUs**: NVIDIA A100 (two separate cards: A100_5, A100_7)
- **Collected data**: top-128 hidden state indices/values and top-128 logprob token IDs/values per generated token
- **Serving**: vLLM with FP8 KV cache
- **Samples**: 1000, **avg sequence length**: 261.6 tokens (min=5, max=3000, median=208)

### Collection protocol

1. **Free generation**: Int8 model on A100_5 generates 1000 responses freely
2. **Enforced generation**: Other model variants replay the same token sequences (enforced decoding) to collect their hidden states and logprobs for the same outputs
3. **Evaluation**: TOPLOC proofs are built from the reference (Int8/A100_5) activations and verified against each variant's activations. All generated tokens are passed to toploc (no position subsampling).

### Proof encoding

Each proof is a polynomial congruence with `k` uint16 coefficients plus a uint16 modulus: **`2 + 2*k` bytes**.
With `decode_batch_size=N`, one proof covers N consecutive tokens.

### Metrics (per sample averages)

- **avg_proofs**: average number of proofs per sample (ceil(sequence_length / batch_size))
- **proof_bytes**: size of a single proof in bytes (`2 + 2*k`)
- **proof_total_bytes**: total proof storage per sample (ceil(avg_proofs) × proof_bytes)
- **bytes/token**: proof storage cost per generated token (proof_total_bytes / avg_sequence_length)
- **exact%**: percentage of proofs with zero exponent mismatches and zero mantissa error
- **exp_mismatch**: mean exponent mismatches per proof
- **mant_err**: mean mantissa error (very large values indicate nearly all exponents differ)

## Int8 (A100_5) vs Int8 (A100_7) — same model, different GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_7 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| hidden_states | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 15 | 17.9 | 34 | 612 | 2.34 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 100.0% | 0.0000 | 0 |
| **logprobs** | **16** | **15** | **17.9** | **34** | **612** | **2.34** | **100.0%** | **0.0000** | **0** |
| logprobs | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 100.0% | 0.0000 | 0 |

## Int8 vs Full Precision — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct` (full precision) on A100_5 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| hidden_states | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 18.1% | 0.0531 | 7.63e+15 |
| hidden_states | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 18.8% | 0.0601 | 1.03e+16 |
| hidden_states | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 19.2% | 0.0628 | 1.09e+16 |
| hidden_states | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 20.3% | 0.0864 | 2.34e+16 |
| hidden_states | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 19.6% | 0.0876 | 9.80e+15 |
| hidden_states | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 3.8% | 0.1006 | 0.8792 |
| hidden_states | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 4.2% | 0.1126 | 0.8481 |
| hidden_states | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 4.6% | 0.1211 | 0.8267 |
| hidden_states | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 5.7% | 0.1709 | 0.7724 |
| hidden_states | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 5.7% | 0.1710 | 0.7772 |
| hidden_states | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 0.2% | 0.1708 | 0.9255 |
| hidden_states | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 0.2% | 0.1877 | 0.8910 |
| hidden_states | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 0.2% | 0.2047 | 0.8715 |
| hidden_states | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 0.7% | 0.2853 | 0.8022 |
| hidden_states | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 1.0% | 0.3181 | 0.8037 |
| hidden_states | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 0.3497 | 1.0308 |
| hidden_states | 16 | 15 | 17.9 | 34 | 612 | 2.34 | 0.0% | 0.3367 | 0.9484 |
| hidden_states | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 0.0% | 0.3340 | 0.9195 |
| hidden_states | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 0.0% | 0.4902 | 0.8367 |
| hidden_states | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 0.0% | 0.5613 | 0.8356 |
| logprobs | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 75.5% | 0.0374 | 1.04e+16 |
| logprobs | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 86.2% | 0.0294 | 6.18e+15 |
| logprobs | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 91.3% | 0.0207 | 5.45e+15 |
| logprobs | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 97.9% | 0.0060 | 0.0421 |
| logprobs | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 98.5% | 0.0048 | 0.0183 |
| logprobs | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 37.6% | 0.0938 | 0.8333 |
| logprobs | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 58.2% | 0.0850 | 0.4220 |
| logprobs | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 71.2% | 0.0771 | 0.2835 |
| logprobs | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 94.5% | 0.0194 | 0.1006 |
| logprobs | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 95.7% | 0.0159 | 0.0600 |
| logprobs | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 4.5% | 0.2776 | 2.2825 |
| logprobs | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 13.8% | 0.2136 | 1.2539 |
| logprobs | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 25.6% | 0.1938 | 0.8145 |
| logprobs | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 85.9% | 0.0629 | 0.2228 |
| logprobs | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 88.4% | 0.0568 | 0.1556 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 1.0433 | 5.5112 |
| **logprobs** | **16** | **15** | **17.9** | **34** | **612** | **2.34** | **0.0%** | **0.7352** | **3.5652** |
| logprobs | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 1.9% | 0.5554 | 2.3734 |
| logprobs | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 58.7% | 0.3358 | 0.4904 |
| logprobs | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 70.4% | 0.2140 | 0.4291 |

## Int8 vs Int4 — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int4` on A100_5 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| hidden_states | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 0.3% | 0.4374 | 8.24e+17 |
| hidden_states | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 0.3% | 0.4662 | 8.88e+17 |
| hidden_states | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 0.2% | 0.4812 | 9.49e+17 |
| hidden_states | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 0.3% | 0.5724 | 1.35e+18 |
| hidden_states | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 0.5% | 0.5884 | 1.30e+18 |
| hidden_states | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 0.0% | 0.8172 | 3.95e+16 |
| hidden_states | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 0.0% | 0.8875 | 5.46e+16 |
| hidden_states | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 0.0% | 0.9291 | 6.26e+16 |
| hidden_states | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 0.0% | 1.1023 | 1.99e+17 |
| hidden_states | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 0.0% | 1.1471 | 8.82e+16 |
| hidden_states | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 0.0% | 1.4186 | 2.08e+15 |
| hidden_states | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 0.0% | 1.4800 | 8.0531 |
| hidden_states | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 0.0% | 1.5625 | 7.7865 |
| hidden_states | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 0.0% | 2.0162 | 6.4762 |
| hidden_states | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 0.0% | 2.0940 | 6.2886 |
| hidden_states | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 2.9004 | 9.0763 |
| hidden_states | 16 | 15 | 17.9 | 34 | 612 | 2.34 | 0.0% | 2.7233 | 8.5730 |
| hidden_states | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 0.0% | 2.6361 | 8.3757 |
| hidden_states | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 0.0% | 3.4863 | 7.0746 |
| hidden_states | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 0.0% | 3.7961 | 6.7680 |
| logprobs | 2 | 10 | 26.6 | 6 | 162 | 0.62 | 47.5% | 0.3793 | 9.50e+17 |
| logprobs | 2 | 15 | 17.9 | 6 | 108 | 0.41 | 63.5% | 0.2926 | 7.35e+17 |
| logprobs | 2 | 20 | 13.5 | 6 | 84 | 0.32 | 73.8% | 0.2136 | 5.50e+17 |
| logprobs | 2 | 100 | 3.1 | 6 | 24 | 0.09 | 95.2% | 0.0292 | 4.69e+16 |
| logprobs | 2 | 200 | 1.9 | 6 | 12 | 0.05 | 96.3% | 0.0255 | 3.92e+16 |
| logprobs | 4 | 10 | 26.6 | 10 | 270 | 1.03 | 14.6% | 0.8976 | 8.95e+16 |
| logprobs | 4 | 15 | 17.9 | 10 | 180 | 0.69 | 27.4% | 0.8385 | 7.63e+16 |
| logprobs | 4 | 20 | 13.5 | 10 | 140 | 0.54 | 39.9% | 0.7364 | 8.31e+16 |
| logprobs | 4 | 100 | 3.1 | 10 | 40 | 0.15 | 89.3% | 0.1242 | 1.17e+16 |
| logprobs | 4 | 200 | 1.9 | 10 | 20 | 0.08 | 91.8% | 0.0982 | 0.1966 |
| logprobs | 8 | 10 | 26.6 | 18 | 486 | 1.86 | 1.6% | 2.0877 | 5.55e+15 |
| logprobs | 8 | 15 | 17.9 | 18 | 324 | 1.24 | 4.6% | 1.8095 | 2.06e+15 |
| logprobs | 8 | 20 | 13.5 | 18 | 252 | 0.96 | 8.6% | 1.7833 | 4.09e+15 |
| logprobs | 8 | 100 | 3.1 | 18 | 72 | 0.28 | 76.2% | 0.4762 | 0.8074 |
| logprobs | 8 | 200 | 1.9 | 18 | 36 | 0.14 | 81.4% | 0.3765 | 0.5763 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 6.7459 | 2.08e+15 |
| **logprobs** | **16** | **15** | **17.9** | **34** | **612** | **2.34** | **0.0%** | **5.1250** | **1.03e+15** |
| logprobs | 16 | 20 | 13.5 | 34 | 476 | 1.82 | 0.7% | 4.1581 | 2.72e+15 |
| logprobs | 16 | 100 | 3.1 | 34 | 136 | 0.52 | 39.3% | 2.3929 | 2.0234 |
| logprobs | 16 | 200 | 1.9 | 34 | 68 | 0.26 | 62.8% | 1.4774 | 1.7122 |

## Baseline: per-token verification (k=20, batch_size=1)

Extreme case: one proof per generated token using all 20 collected logprobs. Same model on two different A100 GPUs.

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| logprobs | 20 | 1 | 261.6 | 42 | 11004 | 42.06 | 100.0% | 0.0000 | 0 |

**261602/261602 proofs exact (100%)** across all 1000 samples — every single token-level proof matches, confirming full reproducibility even at maximum granularity.

## Focused analysis: logprobs k=16, batch_size 8-16

### Focused: Int8 (A100_5) vs Int8 (A100_7) — logprobs k=16

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_7 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| logprobs | 16 | 8 | 33.1 | 34 | 1156 | 4.42 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 12 | 22.3 | 34 | 782 | 2.99 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 14 | 19.1 | 34 | 680 | 2.60 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 16 | 16.8 | 34 | 578 | 2.21 | 100.0% | 0.0000 | 0 |

### Focused: Int8 vs Full Precision — logprobs k=16

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct` (full precision) on A100_5 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| logprobs | 16 | 8 | 33.1 | 34 | 1156 | 4.42 | 0.0% | 1.2078 | 6.5081 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 1.0433 | 5.5112 |
| logprobs | 16 | 12 | 22.3 | 34 | 782 | 2.99 | 0.0% | 0.9196 | 4.6595 |
| logprobs | 16 | 14 | 19.1 | 34 | 680 | 2.60 | 0.0% | 0.7999 | 3.8932 |
| logprobs | 16 | 16 | 16.8 | 34 | 578 | 2.21 | 0.3% | 0.6849 | 3.2864 |

#### False positive analysis (sequence level)

**batch_size=16**: 57 false positive proofs across 33 samples (1 fully accepted)

| sample | seq_len | n_proofs | n_exact | fp_rate |
|-------:|--------:|--------:|--------:|--------:|
| 6 | 349 | 22 | 2 | 9.1% |
| 54 | 338 | 22 | 1 | 4.5% |
| 57 | 394 | 25 | 2 | 8.0% |
| 94 | 584 | 37 | 1 | 2.7% |
| 98 | 338 | 22 | 1 | 4.5% |
| 114 | 493 | 31 | 1 | 3.2% |
| 116 | 689 | 44 | 2 | 4.5% |
| 123 | 409 | 26 | 1 | 3.8% |
| 127 | 426 | 27 | 1 | 3.7% |
| 175 | 395 | 25 | 2 | 8.0% |
| 223 | 1440 | 90 | 2 | 2.2% |
| 280 | 222 | 14 | 1 | 7.1% |
| 357 | 210 | 14 | 1 | 7.1% |
| 383 | 399 | 25 | 1 | 4.0% |
| 411 | 396 | 25 | 1 | 4.0% |
| 442 | 11 | 1 | 1 | 100.0% **(fully accepted)** |
| 443 | 429 | 27 | 9 | 33.3% |
| 444 | 465 | 30 | 2 | 6.7% |
| 447 | 154 | 10 | 3 | 30.0% |
| 566 | 1233 | 78 | 1 | 1.3% |
| 593 | 617 | 39 | 3 | 7.7% |
| 614 | 899 | 57 | 5 | 8.8% |
| 630 | 175 | 11 | 1 | 9.1% |
| 688 | 433 | 28 | 1 | 3.6% |
| 697 | 927 | 58 | 2 | 3.4% |
| 731 | 410 | 26 | 1 | 3.8% |
| 745 | 182 | 12 | 1 | 8.3% |
| 772 | 500 | 32 | 1 | 3.1% |
| 782 | 540 | 34 | 1 | 2.9% |
| 831 | 680 | 43 | 1 | 2.3% |
| 840 | 356 | 23 | 2 | 8.7% |
| 983 | 444 | 28 | 1 | 3.6% |
| 988 | 628 | 40 | 1 | 2.5% |

### Focused: Int8 vs Int4 — logprobs k=16

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int4` on A100_5 (enforced tokens)

| Mode | k | batch_size | avg_proofs | proof_bytes | proof_total_bytes | bytes/token | exact% | exp_mismatch | mant_err |
|------|--:|-----------:|-----------:|------------:|------------------:|------------:|-------:|-------------:|---------:|
| logprobs | 16 | 8 | 33.1 | 34 | 1156 | 4.42 | 0.0% | 7.5328 | 2.78e+15 |
| logprobs | 16 | 10 | 26.6 | 34 | 918 | 3.51 | 0.0% | 6.7459 | 2.08e+15 |
| logprobs | 16 | 12 | 22.3 | 34 | 782 | 2.99 | 0.0% | 6.0765 | 1.66e+15 |
| logprobs | 16 | 14 | 19.1 | 34 | 680 | 2.60 | 0.0% | 5.4198 | 2.89e+15 |
| logprobs | 16 | 16 | 16.8 | 34 | 578 | 2.21 | 0.1% | 4.8665 | 16.1360 |

#### False positive analysis (sequence level)

**batch_size=16**: 10 false positive proofs across 8 samples (0 fully accepted)

| sample | seq_len | n_proofs | n_exact | fp_rate |
|-------:|--------:|--------:|--------:|--------:|
| 6 | 349 | 22 | 1 | 4.5% |
| 223 | 1440 | 90 | 2 | 2.2% |
| 280 | 222 | 14 | 1 | 7.1% |
| 411 | 396 | 25 | 1 | 4.0% |
| 443 | 429 | 27 | 1 | 3.7% |
| 614 | 899 | 57 | 2 | 3.5% |
| 745 | 182 | 12 | 1 | 8.3% |
| 988 | 628 | 40 | 1 | 2.5% |

## Summary

- **Same model, different GPU**: 100% exact match across all configurations. TOPLOC proofs are fully reproducible across A100 cards.
- **Int8 vs Full Precision**: Small but detectable divergence. Hidden states show low exponent mismatch with moderate mantissa error. Larger batch sizes increase exact match rate (especially for logprobs at low k) because more tokens are aggregated into each proof, diluting per-token differences.
- **Int8 vs Int4**: Large divergence across all metrics. Logprob mantissa errors frequently overflow. Hidden states fare better but still show significant mismatch. Even with large batch sizes and low k, int4 divergence remains detectable.
- **False positives at batch_size=16**: A small number of individual proofs pass verification despite model mismatch (57/16828 for int8 vs full, 10/16828 for int8 vs int4). At the sequence level, these are scattered across many samples (1-9 FP proofs out of 10-90 per sample). Only 1 sample (seq_len=11, 1 proof) is fully accepted as a false positive — very short sequences with a single proof are the main risk. No short-sequence false positives occur for int8 vs int4.

## Notes on proof encoding (paper vs library)

The toploc library matches the paper (arXiv:2501.16007v2) on proof encoding:
- Polynomial coefficients are **uint16** (2 bytes each), interpolated in an integer field modulo a prime
- Modulus is **uint16** (searched downward from 65497 to find an injective mapping)
- Proof wire format: **2 bytes (modulus) + k × 2 bytes (coefficients)**
- For k=128 (paper's default): **258 bytes per proof**, covering `batch_size` tokens
- Paper example: 258 bytes per 32 tokens = 8 bytes/token (1024× reduction vs storing full hidden states)
