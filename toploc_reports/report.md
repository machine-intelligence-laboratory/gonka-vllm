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
3. **Evaluation**: TOPLOC proofs are built from the reference (Int8/A100_5) activations and verified against each variant's activations

### Metrics

- **exact**: number of proofs with zero exponent mismatches and zero mantissa error
- **pct**: percentage of exact matches
- **exp_mismatch**: mean number of top-k values where bfloat16 exponents differ between proof and verification
- **mant_err**: mean mantissa error (averaged over matching-exponent values; very large values indicate nearly all exponents differ)
- **k**: number of top activations used in the proof polynomial
- **#pos**: number of evenly-spaced token positions sampled from each response

## Int8 (A100_5) vs Int8 (A100_7) — same model, different GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_7 (enforced tokens)

| Mode | k | #pos | exact | pct | exp_mismatch | mant_err |
|------|--:|-----:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 2 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 4 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 8 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| hidden_states | 16 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 2 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 4 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 8 | 8 | 999/999 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 1 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 2 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 4 | 1000/1000 | 100.0% | 0.0000 | 0 |
| logprobs | 16 | 8 | 999/999 | 100.0% | 0.0000 | 0 |

## Int8 vs Full Precision — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct` (full precision) on A100_5 (enforced tokens)

| Mode | k | #pos | exact | pct | exp_mismatch | mant_err |
|------|--:|-----:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 1 | 162/1000 | 16.2% | 0.0390 | 1.84e+16 |
| hidden_states | 2 | 2 | 185/1000 | 18.5% | 0.0400 | 0.7560 |
| hidden_states | 2 | 4 | 208/1000 | 20.8% | 0.0500 | 0.7475 |
| hidden_states | 2 | 8 | 207/999 | 20.7% | 0.0480 | 0.7878 |
| hidden_states | 4 | 1 | 16/1000 | 1.6% | 0.0650 | 0.8909 |
| hidden_states | 4 | 2 | 19/1000 | 1.9% | 0.0780 | 0.9011 |
| hidden_states | 4 | 4 | 31/1000 | 3.1% | 0.0980 | 0.8725 |
| hidden_states | 4 | 8 | 48/999 | 4.8% | 0.1021 | 0.8184 |
| hidden_states | 8 | 1 | 0/1000 | 0.0% | 0.1780 | 1.1107 |
| hidden_states | 8 | 2 | 0/1000 | 0.0% | 0.1470 | 0.9870 |
| hidden_states | 8 | 4 | 1/1000 | 0.1% | 0.1300 | 0.9614 |
| hidden_states | 8 | 8 | 1/999 | 0.1% | 0.1652 | 0.9291 |
| hidden_states | 16 | 1 | 0/1000 | 0.0% | 0.3340 | 1.3783 |
| hidden_states | 16 | 2 | 0/1000 | 0.0% | 0.2870 | 1.2097 |
| hidden_states | 16 | 4 | 0/1000 | 0.0% | 0.2770 | 1.0785 |
| hidden_states | 16 | 8 | 0/999 | 0.0% | 0.2603 | 1.0009 |
| logprobs | 2 | 1 | 23/1000 | 2.3% | 0.1940 | 3.69e+16 |
| logprobs | 2 | 2 | 283/1000 | 28.3% | 0.0570 | 2.0045 |
| logprobs | 2 | 4 | 531/1000 | 53.1% | 0.0540 | 1.84e+16 |
| logprobs | 2 | 8 | 803/999 | 80.4% | 0.0280 | 0.1802 |
| logprobs | 4 | 1 | 0/1000 | 0.0% | 0.5220 | 12.5623 |
| logprobs | 4 | 2 | 0/1000 | 0.0% | 0.3290 | 7.7608 |
| logprobs | 4 | 4 | 67/1000 | 6.7% | 0.1280 | 2.5949 |
| logprobs | 4 | 8 | 365/999 | 36.5% | 0.0811 | 0.6547 |
| logprobs | 8 | 1 | 0/1000 | 0.0% | 1.2690 | 15.1262 |
| logprobs | 8 | 2 | 0/1000 | 0.0% | 0.9440 | 11.0669 |
| logprobs | 8 | 4 | 0/1000 | 0.0% | 0.6390 | 7.2866 |
| logprobs | 8 | 8 | 6/999 | 0.6% | 0.2943 | 2.7657 |
| logprobs | 16 | 1 | 0/1000 | 0.0% | 2.7180 | 16.4821 |
| logprobs | 16 | 2 | 0/1000 | 0.0% | 2.2220 | 13.2216 |
| logprobs | 16 | 4 | 0/1000 | 0.0% | 1.7600 | 10.3494 |
| logprobs | 16 | 8 | 0/999 | 0.0% | 1.2262 | 7.0174 |

## Int8 vs Int4 — same GPU

Reference: `Qwen2.5-3B-Instruct-GPTQ-Int8` on A100_5 (free generation)
Verify against: `Qwen2.5-3B-Instruct-GPTQ-Int4` on A100_5 (enforced tokens)

| Mode | k | #pos | exact | pct | exp_mismatch | mant_err |
|------|--:|-----:|------:|----:|-------------:|---------:|
| hidden_states | 2 | 1 | 1/1000 | 0.1% | 0.3180 | 4.24e+17 |
| hidden_states | 2 | 2 | 3/1000 | 0.3% | 0.3830 | 5.35e+17 |
| hidden_states | 2 | 4 | 2/1000 | 0.2% | 0.3850 | 4.98e+17 |
| hidden_states | 2 | 8 | 4/999 | 0.4% | 0.3774 | 5.54e+17 |
| hidden_states | 4 | 1 | 0/1000 | 0.0% | 0.4100 | 6.6351 |
| hidden_states | 4 | 2 | 0/1000 | 0.0% | 0.5620 | 1.84e+16 |
| hidden_states | 4 | 4 | 0/1000 | 0.0% | 0.7010 | 3.69e+16 |
| hidden_states | 4 | 8 | 0/999 | 0.0% | 0.8158 | 6.6457 |
| hidden_states | 8 | 1 | 0/1000 | 0.0% | 1.1260 | 7.3628 |
| hidden_states | 8 | 2 | 0/1000 | 0.0% | 1.0240 | 7.6056 |
| hidden_states | 8 | 4 | 0/1000 | 0.0% | 1.0970 | 7.7708 |
| hidden_states | 8 | 8 | 0/999 | 0.0% | 1.4354 | 7.5836 |
| hidden_states | 16 | 1 | 0/1000 | 0.0% | 1.7660 | 8.6900 |
| hidden_states | 16 | 2 | 0/1000 | 0.0% | 2.0980 | 9.0580 |
| hidden_states | 16 | 4 | 0/1000 | 0.0% | 2.2300 | 8.6004 |
| hidden_states | 16 | 8 | 0/999 | 0.0% | 2.2783 | 8.4461 |
| logprobs | 2 | 1 | 2/1000 | 0.2% | 0.8700 | 2.23e+18 |
| logprobs | 2 | 2 | 106/1000 | 10.6% | 0.3950 | 6.46e+17 |
| logprobs | 2 | 4 | 283/1000 | 28.3% | 0.3770 | 8.85e+17 |
| logprobs | 2 | 8 | 518/999 | 51.9% | 0.3433 | 8.68e+17 |
| logprobs | 4 | 1 | 0/1000 | 0.0% | 2.3030 | 1.64e+18 |
| logprobs | 4 | 2 | 0/1000 | 0.0% | 1.8370 | 5.53e+17 |
| logprobs | 4 | 4 | 16/1000 | 1.6% | 0.9660 | 9.22e+16 |
| logprobs | 4 | 8 | 149/999 | 14.9% | 0.7698 | 5.54e+16 |
| logprobs | 8 | 1 | 0/1000 | 0.0% | 5.2930 | 8.30e+17 |
| logprobs | 8 | 2 | 0/1000 | 0.0% | 4.7350 | 2.40e+17 |
| logprobs | 8 | 4 | 0/1000 | 0.0% | 3.7270 | 1.84e+16 |
| logprobs | 8 | 8 | 1/999 | 0.1% | 2.1061 | 14.0402 |
| logprobs | 16 | 1 | 0/1000 | 0.0% | 11.4110 | 3.69e+17 |
| logprobs | 16 | 2 | 0/1000 | 0.0% | 10.6830 | 1.29e+17 |
| logprobs | 16 | 4 | 0/1000 | 0.0% | 9.4800 | 24.7729 |
| logprobs | 16 | 8 | 0/999 | 0.0% | 7.4885 | 21.1297 |

## Summary

- **Same model, different GPU**: 100% exact match across all configurations. TOPLOC proofs are fully reproducible across A100 cards.
- **Int8 vs Full Precision**: Small but detectable divergence. Hidden states show low exponent mismatch (~0.04-0.33) with moderate mantissa error. More sampled positions generally improve exact match rate for logprobs at low k.
- **Int8 vs Int4**: Large divergence across all metrics. Logprob proofs break down severely (mantissa errors overflow). Hidden states fare slightly better but still show significant mismatch, confirming that aggressive quantization fundamentally alters the activation landscape.
