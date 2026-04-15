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

| k | bs | B/tok | FP % (count)    | FN % (count)       | t_exp  | t_mant |
|--:|--:|------:|----------------:|-------------------:|-------:|-------:|
| 512 | 8  | 129.6 | 0.18% (2/1100) | **0.00% (0/1100)** | 0.0†   | 9.926  |
| 512 | 16 | 65.7  | 0.18% (2/1100) | 0.18% (2/1100)     | 71.0   | 9.578  |
| 256 | 8  | 64.9  | 0.18% (2/1100) | 0.55% (6/1100)     | 0.0†   | 10.017 |
| 512 | 32 | 33.7  | 0.18% (2/1100) | 4.55% (50/1100)    | 66.0   | 9.343  |
| 256 | 16 | 32.9  | 0.18% (2/1100) | 5.00% (55/1100)    | 33.75  | 9.184  |
| 128 | 8  | 32.6  | 0.18% (2/1100) | 6.82% (75/1100)    | 0.0†   | 9.913  |
| 256 | 32 | 16.9  | 0.18% (2/1100) | 8.73% (96/1100)    | 0.0†   | 8.700  |
| 64  | 8  | 16.4  | 0.18% (2/1100) | 13.82% (152/1100)  | 7.4    | 9.311  |
| 16  | 32 | 1.12  | 0.18% (2/1100) | 28.45% (313/1100)  | 2.567  | 6.282  |

† `t_exp = 0` → the 2D optimum collapses to mantissa-only: flag iff
`mean_mant > t_mant`. At these k values mantissa alone separates the bulk of
A100 honest runs from INT4 fraud.

### Recommended operating points

- **Strict (FN=0%): k=512, bs=8** → 130 B/token, flag iff `mean_mant > 9.93`.
  Every fraud caught; only the two persistent A100 outliers falsely flagged.
- **Balanced: k=512, bs=16** → 66 B/token, joint threshold (t_exp=71,
  t_mant=9.58). Symmetric 2-out-of-1100 error on each side.
- **Low-cost: k=128, bs=8** → 33 B/token, `mean_mant > 9.91`. FP=0.18%,
  FN=6.82% (75 fraud samples escape out of 1100).

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
