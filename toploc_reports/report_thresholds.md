# TOPLOC Threshold-Based Cross-Architecture Verification — Qwen3-235B-A22B

Supersedes the pessimistic conclusion of `report_separation.md`. The prior
report used per-sample **max** of positive-control errors as the threshold and
concluded cross-architecture verification is impossible. This report shows
that using per-sample **mean** exp_mismatch and mean mant_err — optionally
joined into a 2D threshold — separates cross-architecture (A100→H100 FP8,
honest) from cross-quantization (INT4→FP8, fraud) with very low false
positive and false negative rates.

## Question

Setting: 1100 matched samples on `Qwen3-235B-A22B-Instruct-2507` (MoE,
235B/22B, hidden_dim=4096, TP=4) with TOPLOC top-512 hidden-state proofs.

- Positive control: A100 FP8 → H100 FP8 (honest cross-architecture verify)
- Positive control: H100 FP8 → A100 FP8 (honest cross-architecture, reverse)
- Negative control: H100 INT4 → H100 FP8 (cross-quantization fraud)

For each (k, batch_size) we ask: can we pick a threshold that rarely flags
the honest positive run while almost always catching the fraud?

## Key insight

The `report_hs_large.md` data showed a clean ~3× gap in **per-sample mean
statistics** between the two regimes (at k=128, bs=128: mean_exp 7 vs 22,
mant_err 3.1 vs 10.1). That gap was masked in the prior report because it
took the **max** of positive errors — a handful of catastrophic A100 proofs
dominate and swamp the separation.

Switching to per-sample means recovers the separation, at the cost of
allowing a tiny number of honest runs to be falsely flagged — specifically,
two consistently-bad A100 prompts.

## Persistent A100 outliers

Across all 49 (k, bs) configurations, exactly two A100 prompts behave as
near-catastrophic (normalized mean_exp/k approaching 1.0):

| Rank | Sample idx (in A100 jsonl) | seq_len | avg(mean_exp / k) across 49 configs |
|-----:|----------------------------:|--------:|------------------------------------:|
| 1    | 692                         | 227     | **0.892** |
| 2    | 605                         | 3152    | **0.755** |
| 3    | 226                         | 154     | 0.173 |
| 4    | 596                         | 80      | 0.161 |
| 5    | 443                         | 90      | 0.160 |

Rank 3 drops sharply to 0.17 — there is a clean gap after the top two. These
two prompts produce hidden states that are numerically indistinguishable from
fraud under the A100/H100 FP8 implementation gap. For honest verification we
treat them as known-bad and budget exactly 2 false positives.

## Result: false positive vs false negative rates (2D joint threshold)

Policy: flag a sample as fraud iff `mean_exp > t_exp` AND `mean_mant > t_mant`.
Thresholds are chosen to admit exactly 2 false positives (the two persistent
outliers above) → FP rate = 2/1100 = **0.18%** in every row.

FN rate = fraction of 1100 INT4 samples whose (mean_exp, mean_mant) falls
below the joint threshold (fraud that escapes detection).

Where `t_exp = 0.0`, the 2D optimum collapses to mantissa-only: flag iff
`mean_mant > t_mant`. At these k values mantissa alone separates the bulk
of cross-arch honest runs from INT4 fraud.

| k | bs | B/tok | FP % (count)    | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 0.18% (2/1100) | **0.00% (0/1100)** | 0.0    | 9.926  |
| 512 | 16 | 65.7  | 0.18% (2/1100) | 0.18% (2/1100)     | 71.0   | 9.578  |
| 256 | 8  | 64.9  | 0.18% (2/1100) | 0.55% (6/1100)     | 0.0    | 10.017 |
| 512 | 32 | 33.7  | 0.18% (2/1100) | 4.55% (50/1100)    | 66.0   | 9.343  |
| 256 | 16 | 32.9  | 0.18% (2/1100) | 5.00% (55/1100)    | 33.75  | 9.184  |
| 128 | 8  | 32.6  | 0.18% (2/1100) | 6.82% (75/1100)    | 0.0    | 9.913  |
| 256 | 32 | 16.9  | 0.18% (2/1100) | 8.73% (96/1100)    | 0.0    | 8.700  |
| 64  | 8  | 16.4  | 0.18% (2/1100) | 13.82% (152/1100)  | 7.4    | 9.311  |
| 16  | 32 | 1.12  | 0.18% (2/1100) | 28.45% (313/1100)  | 2.567  | 6.282  |

### Recommended operating points

- **Strict (FN=0%): k=512, bs=8** → 130 B/token, flag iff `mean_mant > 9.93`.
  Every fraud caught; only the two persistent A100 outliers falsely flagged.
- **Balanced: k=512, bs=16** → 66 B/token, joint threshold (t_exp=71,
  t_mant=9.58). Symmetric 2-out-of-1100 error on each side.
- **Low-cost: k=128, bs=8** → 33 B/token, `mean_mant > 9.91`. FP=0.18%,
  FN=6.82% (75 fraud samples escape out of 1100).

## Trading FP for cheaper proofs (0.45% and 1% FPR budgets)

If we can tolerate a few more honest A100 runs being flagged, the bytes/token
required to reach near-zero FN drops sharply. Below: same 2D joint threshold
policy, but with budget = 5 FPs (≈0.45% FPR) and 11 FPs (≈1% FPR).

### 0.45% FPR (5 false positives out of 1100)

| k | bs | B/tok | FP % (count)   | FN % (count)      | t_exp | t_mant |
|--:|--:|------:|---------------:|------------------:|------:|-------:|
| 512 | 8  | 129.6 | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 9.138  |
| 512 | 16 | 65.7  | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 9.254  |
| 256 | 8  | 64.9  | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 9.162  |
| 512 | 32 | 33.7  | 0.45% (5/1100) | 0.82% (9/1100)    | 0.0    | 8.848  |
| 256 | 16 | 32.9  | 0.45% (5/1100) | 0.91% (10/1100)   | 0.0    | 8.840  |
| 128 | 8  | 32.6  | 0.45% (5/1100) | 1.45% (16/1100)   | 0.0    | 9.005  |
| 256 | 32 | 16.9  | 0.45% (5/1100) | 6.64% (73/1100)   | 0.0    | 8.453  |
| 64  | 8  | 16.4  | 0.45% (5/1100) | 6.45% (71/1100)   | 7.0   | 8.295  |
| 128 | 16 | 16.5  | 0.45% (5/1100) | 8.64% (95/1100)   | 0.0    | 8.759  |
| 128 | 32 | 8.48  | 0.45% (5/1100) | 16.09% (177/1100) | 6.0   | 8.275  |
| 64  | 16 | 8.32  | 0.45% (5/1100) | 16.64% (183/1100) | 7.71  | 7.649  |
| 16  | 32 | 1.12  | 0.45% (5/1100) | 24.45% (269/1100) | 2.567 | 5.672  |

### 1.00% FPR (11 false positives out of 1100)

| k | bs | B/tok | FP % (count)     | FN % (count)      | t_exp | t_mant |
|--:|--:|------:|-----------------:|------------------:|------:|-------:|
| 512 | 8  | 129.6 | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.403  |
| 512 | 16 | 65.7  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.273  |
| 256 | 8  | 64.9  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.457  |
| 512 | 32 | 33.7  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.235  |
| 256 | 16 | 32.9  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.367  |
| 128 | 8  | 32.6  | 1.00% (11/1100) | 0.18% (2/1100)    | 0.0    | 8.611  |
| 256 | 32 | 16.9  | 1.00% (11/1100) | 4.55% (50/1100)   | 25.75 | 7.885  |
| 128 | 16 | 16.5  | 1.00% (11/1100) | 3.73% (41/1100)   | 0.0    | 7.980  |
| 64  | 8  | 16.4  | 1.00% (11/1100) | 3.64% (40/1100)   | 6.7   | 7.825  |
| 128 | 32 | 8.48  | 1.00% (11/1100) | 10.64% (117/1100) | 6.0   | 7.694  |
| 64  | 16 | 8.32  | 1.00% (11/1100) | 10.36% (114/1100) | 5.5   | 7.649  |
| 32  | 16 | 4.22  | 1.00% (11/1100) | 11.36% (125/1100) | 3.0   | 7.158  |
| 32  | 32 | 2.17  | 1.00% (11/1100) | 17.18% (189/1100) | 3.5   | 6.742  |

### What changes as FPR is relaxed

- **At 0.45% FPR**: three configurations now achieve **0% FN** — k=512/bs=8
  (130 B/tok), k=512/bs=16 (66 B/tok), and **k=256/bs=8 (65 B/tok)**. The
  16 B/tok regime collapses from ≥14% FN at 0.18% FPR down to ≈7% FN.
- **At 1.00% FPR**: five configurations achieve **0% FN**, and the cheapest
  is now **k=512/bs=32 at 33.7 B/tok or k=256/bs=16 at 32.9 B/tok** — a 2×
  reduction over the strictest budget for the same perfect detection.
  k=128/bs=8 reaches FN=0.18% at the same 33 B/tok cost.

The qualitative pattern: each 2× relaxation of FPR roughly halves the
bytes/token needed to reach a fixed FN. Whether the trade is worthwhile
depends on how often honest A100 verifiers can tolerate being asked to
re-prove a flagged generation.

## Reverse cross-architecture: H100 → A100 FP/FN tables

The reverse direction (H100 FP8 free → A100 FP8 enforced) uses the same
1100 H100 free-generation samples as reference, verified on A100.
Results in `vllm_logs/results_h100_vs_a100_fp8.json`.

### Distribution symmetry

Error distributions are nearly identical in both directions. At k=128,
bs=128: A100→H100 mean_exp=7.00, H100→A100 mean_exp=6.95; mant_err
3.12 vs 3.08. This holds across all 49 (k, bs) configurations.

The same two prompts (idx=692 and idx=605) are catastrophic outliers in
both directions, with normalized scores 0.911/0.754 (H100→A100) vs
0.892/0.755 (A100→H100).

### H100 → A100: 0.18% FPR (2 false positives out of 1100)

| k | bs | B/tok | FP % (count)    | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 0.18% (2/1100) | **0.00% (0/1100)** | 0.0    | 9.537  |
| 512 | 16 | 65.7  | 0.18% (2/1100) | **0.00% (0/1100)** | 0.0    | 9.293  |
| 256 | 8  | 64.9  | 0.18% (2/1100) | **0.00% (0/1100)** | 0.0    | 9.440  |
| 512 | 32 | 33.7  | 0.18% (2/1100) | 3.36% (37/1100)    | 0.0    | 9.315  |
| 256 | 16 | 32.9  | 0.18% (2/1100) | 3.18% (35/1100)    | 32.0   | 9.015  |
| 128 | 8  | 32.6  | 0.18% (2/1100) | 3.27% (36/1100)    | 14.43  | 9.346  |
| 256 | 32 | 16.9  | 0.18% (2/1100) | 12.18% (134/1100)  | 26.67  | 9.043  |
| 64  | 8  | 16.4  | 0.18% (2/1100) | 16.64% (183/1100)  | 0.0    | 9.694  |

### H100 → A100: 0.45% FPR (5 false positives out of 1100)

| k | bs | B/tok | FP % (count)    | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 9.089  |
| 512 | 16 | 65.7  | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 8.708  |
| 256 | 8  | 64.9  | 0.45% (5/1100) | **0.00% (0/1100)** | 0.0    | 8.775  |
| 512 | 32 | 33.7  | 0.45% (5/1100) | 0.55% (6/1100)     | 58.0   | 8.658  |
| 256 | 16 | 32.9  | 0.45% (5/1100) | 1.91% (21/1100)    | 0.0    | 9.050  |
| 128 | 8  | 32.6  | 0.45% (5/1100) | 2.45% (27/1100)    | 14.43  | 9.212  |
| 256 | 32 | 16.9  | 0.45% (5/1100) | 5.91% (65/1100)    | 22.33  | 8.327  |
| 64  | 8  | 16.4  | 0.45% (5/1100) | 7.00% (77/1100)    | 6.5    | 8.581  |

### H100 → A100: 1.00% FPR (11 false positives out of 1100)

| k | bs | B/tok | FP % (count)     | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|-----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.454  |
| 512 | 16 | 65.7  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.369  |
| 256 | 8  | 64.9  | 1.00% (11/1100) | **0.00% (0/1100)** | 0.0    | 8.365  |
| 512 | 32 | 33.7  | 1.00% (11/1100) | **0.00% (0/1100)** | 49.5   | 8.471  |
| 256 | 16 | 32.9  | 1.00% (11/1100) | 0.09% (1/1100)     | 0.0    | 8.480  |
| 128 | 8  | 32.6  | 1.00% (11/1100) | 0.27% (3/1100)     | 0.0    | 8.678  |
| 256 | 32 | 16.9  | 1.00% (11/1100) | 2.55% (28/1100)    | 22.33  | 7.654  |
| 64  | 8  | 16.4  | 1.00% (11/1100) | 3.64% (40/1100)    | 5.2    | 7.957  |

### Direction comparison at key operating points

H100→A100 shows slightly *better* separation than A100→H100 at low FPR.
At 0.18% FPR: H100→A100 achieves 0% FN at k=512/bs=16 (66 B/tok), while
A100→H100 achieves 0% FN only at k=512/bs=8 (130 B/tok). At 1% FPR,
both directions achieve 0% FN at the same configs (k≥256).

### Union thresholds (covering both directions simultaneously)

When the verifier doesn't know which direction it's checking, we combine
both positive-control sets (2200 samples total). FP budget = 2× per
direction.

#### Union: 0.45% FPR per direction (10 FPs out of 2200)

| k | bs | B/tok | FP % (count)     | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|-----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 0.45% (10/2200) | **0.00% (0/1100)** | 0.0    | 9.138  |
| 512 | 16 | 65.7  | 0.45% (10/2200) | **0.00% (0/1100)** | 0.0    | 8.984  |
| 256 | 8  | 64.9  | 0.45% (10/2200) | **0.00% (0/1100)** | 0.0    | 9.024  |
| 512 | 32 | 33.7  | 0.45% (10/2200) | 1.09% (12/1100)    | 58.0   | 8.848  |
| 256 | 16 | 32.9  | 0.45% (10/2200) | 1.91% (21/1100)    | 0.0    | 9.050  |
| 128 | 8  | 32.6  | 0.45% (10/2200) | 2.45% (27/1100)    | 14.43  | 9.212  |

#### Union: 1.00% FPR per direction (22 FPs out of 2200)

| k | bs | B/tok | FP % (count)     | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|-----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 1.00% (22/2200) | **0.00% (0/1100)** | 0.0    | 8.454  |
| 512 | 16 | 65.7  | 1.00% (22/2200) | **0.00% (0/1100)** | 0.0    | 8.303  |
| 256 | 8  | 64.9  | 1.00% (22/2200) | **0.00% (0/1100)** | 0.0    | 8.444  |
| 512 | 32 | 33.7  | 1.00% (22/2200) | **0.00% (0/1100)** | 0.0    | 8.343  |
| 256 | 16 | 32.9  | 1.00% (22/2200) | **0.00% (0/1100)** | 0.0    | 8.409  |
| 128 | 8  | 32.6  | 1.00% (22/2200) | 0.27% (3/1100)     | 0.0    | 8.638  |

### Threshold cross-applicability

A threshold calibrated on one direction works for the other. At the
0.45% FPR budget (5 FPs), applying the A100→H100 mant_err threshold to
H100→A100 positives yields 3-8 FPs (target was 5), and vice versa.

**Takeaway**: a single mant_err threshold in the 8.3–9.1 range works for
both A100→H100 and H100→A100 verification. There is no need to calibrate
per-direction thresholds.

## Sequence-length separability

The per-sample scatter of mean_exp and mean_mant vs sequence length for five
representative (k, bs) configurations is in
`vllm_logs/separability_by_seqlen.png`:

- Green (H100 FP8 → H100 FP8, same-arch): tight cluster near zero on both
  axes at every seq_len → reference baseline.
- Blue (A100 FP8 → H100 FP8, cross-arch honest): bulk well below the fraud
  cluster at every seq_len; two points at high values — the persistent
  outliers.
- Red (H100 INT4 → H100 FP8, fraud): consistently higher than blue bulk
  across all seq_lens.

Sequence length itself is *not* a strong predictor of false positives: one
persistent outlier is short (seq_len=227), the other is long (seq_len=3152).

## Comparison with 1D-only thresholds (at exactly 2 FPs)

For the strongest config (k=512, bs=8):

| Strategy                                | t_exp  | t_mant | FN % |
|-----------------------------------------|-------:|-------:|-----:|
| 1D per-sample mean_exp                  | 91.93  | –      | 1.09% |
| 1D per-sample mean_mant                 | –      | 9.926  | 0.00% |
| 2D joint                                | 0.0    | 9.926  | 0.00% |

At k ≥ 256 mantissa alone is sufficient. Exp adds separating power only at
small k (k ≤ 64), where mant_err occasionally overflows to sentinel values
and becomes noisier.

## Why the prior `report_separation.md` said 0% detection

It set the threshold at `max(positive_control_errors)` — the worst-case
honest proof. The two persistent A100 outliers (and their tails) have
`mean_exp` and `max_exp` approaching k, which completely dominates the INT4
distribution. Forcing strict 0% FPR therefore set the threshold so high that
no fraud sample exceeded it.

This report relaxes 0% → 0.18% FPR (admit those 2 known-bad samples) and
recovers usable separation.

## Reproduction

### Required data

All paths relative to the repo root (`/Users/nikolai/projects/ml/vllm`).

| Role | File |
|------|------|
| Reference tokens, FP8 H100 free (deduped, 1100 samples) | `vllm_logs/fp8_free_filtered.jsonl` |
| Reference tokens, INT4 H100 free                        | `vllm_logs/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16_int4_4xH100_free.jsonl` |
| Reference tokens, FP8 A100 free                         | `a100_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xA100_free.jsonl` |
| Per-proof results, FP8 vs FP8 (same-arch positive)      | `vllm_logs/results_fp8_vs_fp8_large.json` |
| Per-proof results, INT4 vs FP8 (fraud negative)         | `vllm_logs/results_int4_vs_fp8_large.json` |
| Per-proof results, A100 vs H100 (cross-arch positive)   | `a100_logs/results_a100_vs_h100_fp8.json` |
| Per-proof results, H100 vs A100 (reverse cross-arch)    | `vllm_logs/results_h100_vs_a100_fp8.json` |

Each results JSON is a list of 49 entries (one per `(k, decode_batch_size)`
pair with `k, bs ∈ {8, 16, 32, 64, 128, 256, 512}`). Each entry has a
`per_file` array of per-proof records with fields `exp_mismatches`,
`mant_err_mean`, `mant_err_median`. Per-file proofs are emitted in the same
order as the free-side jsonl — group `ceil(seq_len / bs)` consecutive proofs
per sample.

Mantissa overflow sentinel (≈1e14–1e16) is clipped to `MANT_CLIP = 1e6`
before taking means.

### Scripts

All live in `benchmarks/toploc/` and are runnable with the project venv
(`.venv/bin/python3`).

| Script | Purpose |
|--------|---------|
| `benchmarks/toploc/recalc_separation.py`     | Sweep all 49 (k, bs) × 6 strategies (mean_exp, mean_mant, max_exp, max_mant, median_exp, frac_nonzero_exp) × 4 FPR targets. Writes `separation_recalc.json`. |
| `benchmarks/toploc/separability_by_seqlen.py` | Produces `vllm_logs/separability_by_seqlen.png` (seq_len vs mean_exp and mean_mant scatter for 5 configs × 3 classes) and `vllm_logs/a100_outliers.json` (top 10 A100 mean_exp outliers per config). |
| `benchmarks/toploc/final_separation.py`       | Identifies persistent A100 outliers by averaging `mean_exp / k` across all 49 configs. Reports detection under 1D mean_exp at {0, 0.2, 0.5, 1}% FPR, with and without dropping the two outliers. Also runs exhaustive 2D (mean_exp AND mean_mant) threshold search. |
| `benchmarks/toploc/exactly_2fp.py`            | **The main table above.** Computes 1D mean_exp, 1D mean_mant, and 2D joint threshold that admit exactly 2 false positives (idx 605 and 692). Prints the Pareto frontier of detection vs bytes-per-token. |
| `benchmarks/toploc/fpr_tables.py`             | **The 0.45%/1% FPR tables.** Vectorized 2D joint-threshold search. Sweeps FP budgets {2, 5, 11} across 19 representative (k, bs) configurations. |
| `benchmarks/toploc/add_reverse_direction.py`  | **Reverse cross-arch analysis.** Compares A100→H100 vs H100→A100 distributions, finds H100→A100 outliers, computes union thresholds, and tests cross-applicability of thresholds between directions. |

### Commands

```bash
# Quick sweep across strategies and FPR targets (prints highlight tables,
# writes separation_recalc.json):
.venv/bin/python3 benchmarks/toploc/recalc_separation.py

# Identify persistent outliers, run 2D search at 0%/0.5% FPR:
.venv/bin/python3 benchmarks/toploc/final_separation.py

# Reproduce the main report table (FN at exactly 2 FPs):
.venv/bin/python3 benchmarks/toploc/exactly_2fp.py

# Render the separability-by-seq-length figure:
.venv/bin/python3 benchmarks/toploc/separability_by_seqlen.py

# Reproduce the 0.45% / 1.00% FPR tables:
.venv/bin/python3 benchmarks/toploc/fpr_tables.py

# Reverse direction (H100→A100) and threshold symmetry analysis:
.venv/bin/python3 benchmarks/toploc/add_reverse_direction.py
```

### Persistent outliers for pre-filtering

Hard-coded in `benchmarks/toploc/exactly_2fp.py` as
`PERSISTENT_OUTLIERS = {605, 692}`. These indices refer to the row order in
`a100_logs/Qwen3-235B-A22B-Instruct-2507-FP8_fp8_4xA100_free.jsonl`. If the
jsonl is regenerated with a different ordering, re-run
`final_separation.py` to rediscover them — look for the rank-1 and rank-2
entries by `avg(mean_exp / k)`.

### Data cleaning note

The FP8-free H100 collection produced 1354 NPZ entries due to a
collection-protocol change mid-run. `fp8_free_filtered.jsonl` keeps the 1100
new-format entries aligned with the evaluation; the raw 1354-row file
should not be used as the reference side.

## Related files

- `report_hs_large.md` — raw per-config means for all 49 (k, bs) settings on
  the three experiments.
- `report_separation.md` — prior (pessimistic) analysis using max-of-positive
  thresholds.
- `vllm_logs/separability_by_seqlen.png` — main figure.
- `vllm_logs/a100_outliers.json` — per-config top-10 A100 outliers.
- `separation_recalc.json` — all-strategy × all-FPR sweep (written by
  `recalc_separation.py`).
