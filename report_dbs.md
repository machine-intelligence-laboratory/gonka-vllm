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

### Collection protocol

1. **Free generation**: Int8 model on A100_5 generates 1000 responses freely
2. **Enforced generation**: Other model variants replay the same token sequences (enforced decoding) to collect their hidden states and logprobs for the same outputs
3. **Evaluation**: TOPLOC proofs are built from the reference (Int8/A100_5) activations and verified against each variant's activations. All generated tokens are passed to toploc (no position subsampling).

### Metrics

- **exact**: number of proofs with zero exponent mismatches and zero mantissa error
- **pct**: percentage of exact matches
- **exp_mismatch**: mean number of top-k values where bfloat16 exponents differ between proof and verification
- **mant_err**: mean mantissa error (averaged over matching-exponent values; very large values indicate nearly all exponents differ)
- **k**: number of top activations used in the proof polynomial
- **batch_size**: `decode_batching_size` — number of consecutive tokens grouped into one proof. batch_size=1 means one proof per token; larger values batch multiple tokens into a single proof

## Int8 (A100_5) vs Int8 (A100_7) — same model, different GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_7 (enforced tokens)

| Mode | k | batch_size | exact | pct | exp_mismatch | mant_err |
|------|--:|-----------:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 15 | 17901/17901 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 10 | 26596/26596 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 20 | 13546/13546 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 100 | 3148/3148 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 200 | 1883/1883 | 100.0% | 0.0000 | 0 |

## Int8 vs Full Precision — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct` (full precision) on A100_5 (enforced tokens)

| Mode | k | batch_size | exact | pct | exp_mismatch | mant_err |
|------|--:|-----------:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 15 | 3359/17901 | 18.8% | 0.0601 | 1.03e+16 |
| hidden_states | 2 | 10 | 4822/26596 | 18.1% | 0.0531 | 7.63e+15 |
| hidden_states | 2 | 20 | 2601/13546 | 19.2% | 0.0628 | 1.09e+16 |
| hidden_states | 2 | 100 | 639/3148 | 20.3% | 0.0864 | 2.34e+16 |
| hidden_states | 2 | 200 | 370/1883 | 19.6% | 0.0876 | 9.80e+15 |
| hidden_states | 4 | 15 | 748/17901 | 4.2% | 0.1126 | 0.8481 |
| hidden_states | 4 | 10 | 1016/26596 | 3.8% | 0.1006 | 0.8792 |
| hidden_states | 4 | 20 | 627/13546 | 4.6% | 0.1211 | 0.8267 |
| hidden_states | 4 | 100 | 180/3148 | 5.7% | 0.1709 | 0.7724 |
| hidden_states | 4 | 200 | 107/1883 | 5.7% | 0.1710 | 0.7772 |
| hidden_states | 8 | 15 | 37/17901 | 0.2% | 0.1877 | 0.8910 |
| hidden_states | 8 | 10 | 43/26596 | 0.2% | 0.1708 | 0.9255 |
| hidden_states | 8 | 20 | 33/13546 | 0.2% | 0.2047 | 0.8715 |
| hidden_states | 8 | 100 | 21/3148 | 0.7% | 0.2853 | 0.8022 |
| hidden_states | 8 | 200 | 19/1883 | 1.0% | 0.3181 | 0.8037 |
| hidden_states | 16 | 15 | 0/17901 | 0.0% | 0.3367 | 0.9484 |
| hidden_states | 16 | 10 | 0/26596 | 0.0% | 0.3497 | 1.0308 |
| hidden_states | 16 | 20 | 0/13546 | 0.0% | 0.3340 | 0.9195 |
| hidden_states | 16 | 100 | 0/3148 | 0.0% | 0.4902 | 0.8367 |
| hidden_states | 16 | 200 | 0/1883 | 0.0% | 0.5613 | 0.8356 |
| logprobs | 2 | 15 | 15432/17901 | 86.2% | 0.0294 | 6.18e+15 |
| logprobs | 2 | 10 | 20094/26596 | 75.5% | 0.0374 | 1.04e+16 |
| logprobs | 2 | 20 | 12374/13546 | 91.3% | 0.0207 | 5.45e+15 |
| logprobs | 2 | 100 | 3081/3148 | 97.9% | 0.0060 | 0.0421 |
| logprobs | 2 | 200 | 1855/1883 | 98.5% | 0.0048 | 0.0183 |
| logprobs | 4 | 15 | 10423/17901 | 58.2% | 0.0850 | 0.4220 |
| logprobs | 4 | 10 | 10013/26596 | 37.6% | 0.0938 | 0.8333 |
| logprobs | 4 | 20 | 9651/13546 | 71.2% | 0.0771 | 0.2835 |
| logprobs | 4 | 100 | 2974/3148 | 94.5% | 0.0194 | 0.1006 |
| logprobs | 4 | 200 | 1802/1883 | 95.7% | 0.0159 | 0.0600 |
| logprobs | 8 | 15 | 2479/17901 | 13.8% | 0.2136 | 1.2539 |
| logprobs | 8 | 10 | 1197/26596 | 4.5% | 0.2776 | 2.2825 |
| logprobs | 8 | 20 | 3466/13546 | 25.6% | 0.1938 | 0.8145 |
| logprobs | 8 | 100 | 2704/3148 | 85.9% | 0.0629 | 0.2228 |
| logprobs | 8 | 200 | 1664/1883 | 88.4% | 0.0568 | 0.1556 |
| logprobs | 16 | 15 | 8/17901 | 0.0% | 0.7352 | 3.5652 |
| logprobs | 16 | 10 | 0/26596 | 0.0% | 1.0433 | 5.5112 |
| logprobs | 16 | 20 | 260/13546 | 1.9% | 0.5554 | 2.3734 |
| logprobs | 16 | 100 | 1848/3148 | 58.7% | 0.3358 | 0.4904 |
| logprobs | 16 | 200 | 1325/1883 | 70.4% | 0.2140 | 0.4291 |

## Int8 vs Int4 — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int4` on A100_5 (enforced tokens)

| Mode | k | batch_size | exact | pct | exp_mismatch | mant_err |
|------|--:|-----------:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 15 | 49/17901 | 0.3% | 0.4662 | 8.88e+17 |
| hidden_states | 2 | 10 | 73/26596 | 0.3% | 0.4374 | 8.24e+17 |
| hidden_states | 2 | 20 | 32/13546 | 0.2% | 0.4812 | 9.49e+17 |
| hidden_states | 2 | 100 | 10/3148 | 0.3% | 0.5724 | 1.35e+18 |
| hidden_states | 2 | 200 | 10/1883 | 0.5% | 0.5884 | 1.30e+18 |
| hidden_states | 4 | 15 | 0/17901 | 0.0% | 0.8875 | 5.46e+16 |
| hidden_states | 4 | 10 | 1/26596 | 0.0% | 0.8172 | 3.95e+16 |
| hidden_states | 4 | 20 | 0/13546 | 0.0% | 0.9291 | 6.26e+16 |
| hidden_states | 4 | 100 | 0/3148 | 0.0% | 1.1023 | 1.99e+17 |
| hidden_states | 4 | 200 | 0/1883 | 0.0% | 1.1471 | 8.82e+16 |
| hidden_states | 8 | 15 | 0/17901 | 0.0% | 1.4800 | 8.0531 |
| hidden_states | 8 | 10 | 0/26596 | 0.0% | 1.4186 | 2.08e+15 |
| hidden_states | 8 | 20 | 0/13546 | 0.0% | 1.5625 | 7.7865 |
| hidden_states | 8 | 100 | 0/3148 | 0.0% | 2.0162 | 6.4762 |
| hidden_states | 8 | 200 | 0/1883 | 0.0% | 2.0940 | 6.2886 |
| hidden_states | 16 | 15 | 0/17901 | 0.0% | 2.7233 | 8.5730 |
| hidden_states | 16 | 10 | 0/26596 | 0.0% | 2.9004 | 9.0763 |
| hidden_states | 16 | 20 | 0/13546 | 0.0% | 2.6361 | 8.3757 |
| hidden_states | 16 | 100 | 0/3148 | 0.0% | 3.4863 | 7.0746 |
| hidden_states | 16 | 200 | 0/1883 | 0.0% | 3.7961 | 6.7680 |
| logprobs | 2 | 15 | 11359/17901 | 63.5% | 0.2926 | 7.35e+17 |
| logprobs | 2 | 10 | 12639/26596 | 47.5% | 0.3793 | 9.50e+17 |
| logprobs | 2 | 20 | 10002/13546 | 73.8% | 0.2136 | 5.50e+17 |
| logprobs | 2 | 100 | 2997/3148 | 95.2% | 0.0292 | 4.69e+16 |
| logprobs | 2 | 200 | 1813/1883 | 96.3% | 0.0255 | 3.92e+16 |
| logprobs | 4 | 15 | 4907/17901 | 27.4% | 0.8385 | 7.63e+16 |
| logprobs | 4 | 10 | 3877/26596 | 14.6% | 0.8976 | 8.95e+16 |
| logprobs | 4 | 20 | 5408/13546 | 39.9% | 0.7364 | 8.31e+16 |
| logprobs | 4 | 100 | 2811/3148 | 89.3% | 0.1242 | 1.17e+16 |
| logprobs | 4 | 200 | 1729/1883 | 91.8% | 0.0982 | 0.1966 |
| logprobs | 8 | 15 | 825/17901 | 4.6% | 1.8095 | 2.06e+15 |
| logprobs | 8 | 10 | 414/26596 | 1.6% | 2.0877 | 5.55e+15 |
| logprobs | 8 | 20 | 1158/13546 | 8.6% | 1.7833 | 4.09e+15 |
| logprobs | 8 | 100 | 2400/3148 | 76.2% | 0.4762 | 0.8074 |
| logprobs | 8 | 200 | 1533/1883 | 81.4% | 0.3765 | 0.5763 |
| logprobs | 16 | 15 | 0/17901 | 0.0% | 5.1250 | 1.03e+15 |
| logprobs | 16 | 10 | 0/26596 | 0.0% | 6.7459 | 2.08e+15 |
| logprobs | 16 | 20 | 100/13546 | 0.7% | 4.1581 | 2.72e+15 |
| logprobs | 16 | 100 | 1237/3148 | 39.3% | 2.3929 | 2.0234 |
| logprobs | 16 | 200 | 1183/1883 | 62.8% | 1.4774 | 1.7122 |

## Summary

- **Same model, different GPU**: 100% exact match across all configurations. TOPLOC proofs are fully reproducible across A100 cards.
- **Int8 vs Full Precision**: Small but detectable divergence. Hidden states show low exponent mismatch with moderate mantissa error. Larger batch sizes increase exact match rate (especially for logprobs at low k) because more tokens are aggregated into each proof, diluting per-token differences.
- **Int8 vs Int4**: Large divergence across all metrics. Logprob mantissa errors frequently overflow. Hidden states fare better but still show significant mismatch. Even with large batch sizes and low k, int4 divergence remains detectable.
