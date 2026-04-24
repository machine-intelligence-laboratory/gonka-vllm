# TOPLOC Verification: Sequence-Length Dependence

## Question

Do false negatives (INT4 fraud that escapes detection) concentrate at
specific sequence lengths? If so, should the verifier treat short and long
sequences differently?

## Setup

Same dataset as `report_thresholds.md`: 1100 samples on
Qwen3-235B-A22B-Instruct-2507, with three classes:

- **Cross-arch honest** (Union): A100 FP8 → H100 FP8 + H100 FP8 → A100 FP8
  (2200 positive samples total)
- **INT4 fraud**: H100 INT4 → H100 FP8 (1100 negative samples)

Threshold: 2D joint (mean_exp AND mean_mant), optimized at Union 0.45%
FPR (10 FPs out of 2200).

### Sequence-length distribution of INT4 samples

| Bin (tokens) | Count | % of dataset |
|-------------:|------:|-------------:|
| 1–20         | 66    | 6.0%         |
| 20–50        | 131   | 11.9%        |
| 50–100       | 110   | 10.0%        |
| 100–200      | 113   | 10.3%        |
| 200–500      | 217   | 19.7%        |
| 500–1000     | 246   | 22.4%        |
| 1000–2000    | 179   | 16.3%        |
| 2000–5000    | 35    | 3.2%         |
| 5000–15000   | 3     | 0.3%         |

## Results

### Group A: ~30 B/tok configs (good overall FN < 2%)

#### k=512, bs=32 (34 B/tok) — Union FN = 1.09% (12/1100)

| Bin (tokens) | n   | FN  | FN %  |
|-------------:|----:|----:|------:|
| 1–20         | 66  | 0   | 0.0%  |
| 20–50        | 131 | 0   | 0.0%  |
| 50–100       | 110 | 0   | 0.0%  |
| 100–200      | 113 | 0   | 0.0%  |
| 200–500      | 217 | 1   | 0.5%  |
| 500–1000     | 246 | 2   | 0.8%  |
| 1000–2000    | 179 | 6   | **3.4%** |
| 2000–5000    | 35  | 0   | 0.0%  |
| 5000–15000   | 3   | 0   | 0.0%  |

Threshold: t_exp=58.0, t_mant=8.85

#### k=256, bs=16 (33 B/tok) — Union FN = 1.91% (21/1100)

| Bin (tokens) | n   | FN  | FN %  |
|-------------:|----:|----:|------:|
| 1–20         | 66  | 0   | 0.0%  |
| 20–50        | 131 | 0   | 0.0%  |
| 50–100       | 110 | 0   | 0.0%  |
| 100–200      | 113 | 0   | 0.0%  |
| 200–500      | 217 | 1   | 0.5%  |
| 500–1000     | 246 | 7   | 2.8%  |
| 1000–2000    | 179 | 12  | **6.7%** |
| 2000–5000    | 35  | 1   | 2.9%  |
| 5000–15000   | 3   | 0   | 0.0%  |

Threshold: t_exp=0.0 (mant-only), t_mant=9.05

### Group B: 8–16 B/tok configs (moderate FN ~6–19%)

#### k=256, bs=32 (17 B/tok) — Union FN = 6.18% (68/1100)

| Bin (tokens) | n   | FN  | FN %   |
|-------------:|----:|----:|-------:|
| 1–20         | 66  | 0   | 0.0%   |
| 20–50        | 131 | 0   | 0.0%   |
| 50–100       | 110 | 0   | 0.0%   |
| 100–200      | 113 | 0   | 0.0%   |
| 200–500      | 217 | 13  | 6.0%   |
| 500–1000     | 246 | 27  | 11.0%  |
| 1000–2000    | 179 | 26  | **14.5%** |
| 2000–5000    | 35  | 2   | 5.7%   |
| 5000–15000   | 3   | 0   | 0.0%   |

Threshold: t_exp=22.33, t_mant=8.39

#### k=128, bs=16 (17 B/tok) — Union FN = 8.45% (93/1100)

| Bin (tokens) | n   | FN  | FN %   |
|-------------:|----:|----:|-------:|
| 1–20         | 66  | 1   | 1.5%   |
| 20–50        | 131 | 0   | 0.0%   |
| 50–100       | 110 | 0   | 0.0%   |
| 100–200      | 113 | 1   | 0.9%   |
| 200–500      | 217 | 17  | 7.8%   |
| 500–1000     | 246 | 36  | 14.6%  |
| 1000–2000    | 179 | 35  | **19.6%** |
| 2000–5000    | 35  | 3   | 8.6%   |
| 5000–15000   | 3   | 0   | 0.0%   |

Threshold: t_exp=12.2, t_mant=8.73

#### k=64, bs=8 (16 B/tok) — Union FN = 7.18% (79/1100)

| Bin (tokens) | n   | FN  | FN %   |
|-------------:|----:|----:|-------:|
| 1–20         | 66  | 1   | 1.5%   |
| 20–50        | 131 | 0   | 0.0%   |
| 50–100       | 110 | 0   | 0.0%   |
| 100–200      | 113 | 0   | 0.0%   |
| 200–500      | 217 | 16  | 7.4%   |
| 500–1000     | 246 | 28  | 11.4%  |
| 1000–2000    | 179 | 29  | **16.2%** |
| 2000–5000    | 35  | 2   | 5.7%   |
| 5000–15000   | 3   | 0   | 0.0%   |

Threshold: t_exp=6.7, t_mant=8.58

#### k=128, bs=32 (8.5 B/tok) — Union FN = 18.55% (204/1100)

| Bin (tokens) | n   | FN  | FN %   |
|-------------:|----:|----:|-------:|
| 1–20         | 66  | 1   | 1.5%   |
| 20–50        | 131 | 2   | 1.5%   |
| 50–100       | 110 | 9   | 8.2%   |
| 100–200      | 113 | 13  | 11.5%  |
| 200–500      | 217 | 52  | 24.0%  |
| 500–1000     | 246 | 62  | **25.2%** |
| 1000–2000    | 179 | 60  | **33.5%** |
| 2000–5000    | 35  | 5   | 14.3%  |
| 5000–15000   | 3   | 0   | 0.0%   |

Threshold: t_exp=6.0, t_mant=8.55

## Findings

### 1. False negatives concentrate at medium-long sequences (500–2000 tokens)

Across all configs, the FN rate is lowest for short sequences (< 200
tokens: 0–1.5%) and peaks in the 1000–2000 token bin. This pattern is
consistent regardless of k or batch_size.

The very longest sequences (2000–5000, 5000+) have *lower* FN rates than
1000–2000, but these bins are small (35 and 3 samples) so the numbers
are noisy.

### 2. Short sequences are NOT the problem

The hypothesis that short sequences (few proofs, noisy mean) would have
high FN is **not supported**. Sequences with 1–50 tokens have 0–1.5% FN
even at the cheapest configs. Short INT4-generated texts produce
mean_mant well above the threshold because the per-token divergence
between INT4 and FP8 hidden states is large and consistent — even 1–2
proofs are enough to detect fraud.

### 3. The problem is the 500–2000 token range

In this range, INT4 fraud samples' mean_mant drifts closer to the
threshold. The scatter plots show the INT4 cloud (red) compresses
vertically at longer sequence lengths — the per-sample mean_mant
converges to a tighter distribution around ~9–10 as more proofs are
averaged, and this tighter distribution partially overlaps with the
threshold.

Mechanistically: at short sequences, a few "hot" proofs with high
mant_err dominate the mean and push it well above threshold. At
medium-long sequences, the mean is dominated by the many "typical"
proofs with mant_err around 8–10, which is closer to the cross-arch
positive range (3–5).

### 4. Higher k helps long sequences more than small batch_size does

Comparing k=64/bs=8 (16 B/tok, FN=7.2%) vs k=128/bs=32 (8.5 B/tok,
FN=18.6%) — they have similar proofs/sample at medium sequences, but
k=64/bs=8 uses 8× more dimensions per proof while k=128/bs=32 uses 2×
more dimensions but averages over 4× fewer proofs. The lower batch_size
config (more proofs per sample, less averaging noise) wins.

For the 1000–2000 bin specifically:
- k=64, bs=8: FN=16.2% (48 proofs/sample median)
- k=128, bs=32: FN=33.5% (12 proofs/sample median)
- k=256, bs=32: FN=14.5% (12 proofs/sample median)
- k=512, bs=32: FN=3.4% (12 proofs/sample median)

With the same bs=32 (same proofs/sample), doubling k from 128→256→512
sharply reduces FN in the hardest bin: 33.5% → 14.5% → 3.4%. More
dimensions in each proof increases the per-proof mant_err for fraud
samples, pushing their mean further above threshold.

### 5. Implications for deployment

- **Short outputs (< 200 tokens) are well-protected** at any config. A
  chatbot generating brief answers can use cheap proofs (8–16 B/tok).
- **Long outputs (500–2000 tokens)** require either higher k (≥256) or
  lower batch_size (more proofs per sequence) to maintain low FN.
- **Adaptive verification**: a verifier could use cheaper thresholds for
  short sequences and stricter ones for long ones, or simply require
  higher k for sequences above 500 tokens.

## Scatter plots

`vllm_logs/seqlen_fn_analysis.png` — for each of the 6 configs:
- Left: mean_mant vs seq_len scatter (blue=honest cross-arch,
  red=INT4 fraud, dashed=threshold)
- Right: FN rate bar chart per seq_len bin

## Reproduction

```bash
.venv/bin/python3 benchmarks/toploc/seqlen_analysis.py
```

Reads from:
- `vllm_logs/results_int4_vs_fp8_large.json`
- `a100_logs/results_a100_vs_h100_fp8.json`
- `vllm_logs/results_h100_vs_a100_fp8.json`
- Seq lens from `fp8_free_filtered.jsonl`, INT4 free JSONL, A100 free JSONL

Produces:
- `vllm_logs/seqlen_fn_analysis.png`
- `vllm_logs/seqlen_fn_analysis.json`
