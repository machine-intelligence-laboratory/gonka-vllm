# TOPLOC Production Validation Proposal — Large-Scale Hidden-State Verification

## Goal

Validate TOPLOC hidden-state hashing at production scale: large MoE model (235B total / 22B active), long contexts, multi-GPU tensor and pipeline parallelism, and workloads representative of real-world use. The previous experiments (Qwen2.5 3B on single A100s, avg 261 tokens) confirmed the method works in principle; this experiment answers whether it holds under production conditions.

## Serving

Our own vLLM fork with TOPLOC instrumentation (top-k hidden-state collection hook on the last hidden layer). PagedAttention + FP8 KV cache.

## Model

### Qwen3-235B-A22B-Instruct-2507-FP8

| Parameter | Value |
|-----------|-------|
| Total parameters | 235B |
| Active parameters/token | 22B (8 of 128 experts) |
| Hidden dim | 4,096 |
| Layers | 94 (every layer is MoE) |
| Attention heads / KV heads | 64 / 4 (GQA 16:1) |
| Head dim | 128 |
| Context length | 256K |
| Vocab size | 151,936 |
| Precision | FP8 (weights) + FP8 KV cache |
| MoE expert intermediate size | 1,536 |

**MoE implications for TOPLOC.** The paper did not test MoE models — dense models only (Llama, Intellect-1, Gemma). MoE adds a source of nondeterminism beyond what was characterized:

- **Expert routing**: which 8 of 128 experts are selected per token may vary across runs due to floating-point nondeterminism in the gating network. If two runs route the same token to different experts, the hidden state after the MoE layer will diverge substantially — even on the same hardware.
- **Why this still matters**: the last hidden layer activation (what TOPLOC hashes) sits *after* all 94 MoE layers. If routing is reproducible (same input → same gating decisions), then the hidden state should be reproducible. The experiment will directly test this assumption.
- **Risk**: if MoE routing nondeterminism causes sporadic hidden-state divergence even for same-model/same-hardware, the positive-control experiment (Phase 1) will reveal it immediately. This would be a novel finding not covered by the paper.

**Why this model**: it is what we actually serve in production for text workloads. Testing on the actual production model makes the results directly actionable, rather than extrapolating from a different architecture.

### INT4 model for cross-quantization comparison

To validate that TOPLOC can distinguish between different quantizations of the same model, we need an INT4 variant of Qwen3-235B-A22B-Instruct-2507. No official INT4 instruct model exists from Qwen. Available options:

| Option | Model | Publisher | vLLM support | Instruct | Notes |
|--------|-------|-----------|:------------:|:--------:|-------|
| A | [Qwen3-235B-A22B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4) | Qwen (official) | Yes | No | Official quantization but **base model only** — won't follow instructions, test set would need redesign around completion-style prompts. TOPLOC proof mechanics are identical; loses production-representativeness. |
| B | [Qwen3-235B-A22B-Instruct-2507-INT4-W4A16](https://huggingface.co/chriswritescode/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16) | chriswritescode (individual) | Yes (>=0.8.5) | Yes | Community quantization by a single individual. INT4 weights / FP16 activations. No official accuracy benchmarks. Non-thinking mode only. |
| C | [Qwen3-235B-A22B-Instruct-2507-int4-mixed-AutoRound](https://huggingface.co/Intel/Qwen3-235B-A22B-Instruct-2507-int4-mixed-AutoRound) | Intel | Yes | Yes | Mixed precision: 4-bit experts, 8-bit non-expert layers, 16-bit gate layers (gate kept at 16-bit for vLLM compatibility). Uses AutoRound (RTN, group_size=64, symmetric). No accuracy benchmarks published. |

**Ruled out:**
- [nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4) — NVIDIA FP4, excellent accuracy (benchmarks within 1% of BF16), but requires TensorRT-LLM, not vLLM.
- [lmstudio-community/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/lmstudio-community/Qwen3-235B-A22B-Instruct-2507-GGUF) — GGUF is llama.cpp's binary format (available in 3/4/6/8-bit), not compatible with vLLM.
- [lmstudio-community/Qwen3-235B-A22B-Instruct-2507-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3-235B-A22B-Instruct-2507-MLX-4bit) — MLX format, Apple Silicon only.

## Hardware Grid

All experiments run on the following setups, all using the same vLLM fork:

| Setup | GPUs | Parallelism | Notes |
|-------|------|-------------|-------|
| 4×H100 | 4× H100 80GB | TP=4 | Primary reference |
| 4×B200 | 4× B200 | TP=4 | Different GPU architecture |
| 4×A100 | 4× A100 80GB | TP=4 | Older GPU generation |
| 8×H100 | 8× H100 80GB | TP=4, PP=2 | Pipeline parallelism added |

This gives us 6 cross-hardware comparison pairs at TP=4, plus the TP=4 vs TP=4/PP=2 comparison (4×H100 vs 8×H100).

## Hyperparameter Grid

### k (top-k hidden-state values)

Hidden dim is 4,096 — the same as Llama 3.1-8B from the paper. The paper's k ratios apply directly:

| k | % of hidden dim | Rationale |
|--:|----------------:|-----------|
| 64 | 1.56% | Minimal footprint |
| 128 | 3.12% | Paper's default; demonstrated low mismatch rate in Figure 3 |
| 256 | 6.25% | Extra margin for MoE-induced noise |
| 512 | 12.5% | Conservative; guards against MoE routing jitter |

From Figure 3 (paper): on Llama 3.1-8B (same hidden dim), top-k index mismatch drops below 2% at k=128 and approaches 0 by k=512. **k=128 is the starting point; k=256 is the expected sweet spot** if MoE adds noise beyond what the paper observed.

Our Qwen2.5 3B results showed k=2 gave 100% exact match for same-model cross-GPU. But that was dense — MoE may require higher k for the positive control to pass.

### batch_size (tokens per proof)

| batch_size | Tokens/proof | Rationale |
|-----------:|-------------:|-----------|
| 16 | 16 | Fine-grained; higher storage, best per-segment detection |
| 32 | 32 | Paper's default |
| 64 | 64 | Practical for long outputs |
| 128 | 128 | Aggressive compression; tests whether dilution kills detection |
| 256 | 256 | Extreme case for very long generations |

**Storage budget at key operating points** (proof = 2 + 2k bytes):

| k | batch_size | bytes/token | Per 10K tokens | Per 100K tokens |
|--:|-----------:|------------:|---------------:|----------------:|
| 128 | 32 | 8.06 | 80.6 KB | 806 KB |
| 128 | 64 | 4.03 | 40.3 KB | 403 KB |
| 256 | 32 | 16.06 | 160.6 KB | 1.6 MB |
| 256 | 64 | 8.03 | 80.3 KB | 803 KB |
| 256 | 128 | 4.02 | 40.2 KB | 402 KB |
| 512 | 64 | 16.03 | 160.3 KB | 1.6 MB |
| 512 | 128 | 8.02 | 80.2 KB | 802 KB |

Target: **< 1 MB per 100K tokens** → k=128/bs=32, k=256/bs=64, or k=512/bs=128.

## Test Set

### Existing prompts (1000 samples)

The current 1000 prompts (Bactrian-X, 200 per language: en/es/zh/hi/ar, avg ~261 output tokens, max 3000). These cover the short-output regime well. Stored in `benchmarks/toploc/prompts.json`.

### Long-output prompts (100 samples, new)

To stress-test TOPLOC at high token positions (10K–30K), we add 100 prompts that naturally elicit long outputs. These are drawn from a single source for simplicity.

**Source**: [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) (same source as the existing 1000 prompts). We select 100 substantive English instructions and wrap each with a system prompt requesting an extremely detailed, comprehensive response. Run with `max_new_tokens=32768`.

Script: `benchmarks/toploc/prepare_long_prompts.py`

| Subset | N | Expected output length | Purpose |
|--------|--:|----------------------:|---------|
| Existing short | 1000 | 200–3,000 tokens | Baseline regime, multilingual |
| New long-output | 100 | 10,000–30,000 tokens | Stress-test error accumulation at high token positions |
| **Total** | **1100** | | |

Not all long prompts will hit 30K — output length depends on the topic and model behavior. We select prompts that ask for writing, explanation, or analysis (naturally longer outputs) and the system prompt encourages comprehensive, extensive responses.

## Experiment Matrix

### Phase 1: Same-model cross-hardware (positive control)

Confirm TOPLOC hidden-state proofs are reproducible across different GPU types and parallelism configs, all running the same Qwen3-235B-A22B-FP8 model.

**Reference**: 4×H100, TP=4 (free generation)

| Validation setup | Parallelism | What it tests |
|------------------|-------------|---------------|
| 4×H100 (same node) | TP=4 | Sanity — self-consistency |
| 4×H100 (different node) | TP=4 | Cross-node, same GPU type |
| 4×B200 | TP=4 | Different GPU architecture (Blackwell vs Hopper) |
| 4×A100 | TP=4 | Older GPU generation (Ampere vs Hopper) |
| 8×H100 | TP=4, PP=2 | Pipeline parallelism — activations computed on different GPUs for different layers |

**Expected**: 100% exact match for same-GPU-type/same-TP, based on our Qwen 3B cross-GPU result. Cross-architecture (H100 vs B200 vs A100) and PP=2 are the unknowns — the paper showed cross-GPU (A100 vs 4090) worked, but different architectures and pipeline parallelism were not tested.

**MoE-specific risk**: if gating decisions differ across GPU types due to FP precision differences, the positive control may fail on cross-architecture pairs. This would be a key finding.

### Phase 2: Cross-hardware detection separation

For each hardware pair where Phase 1 shows proof match, confirm that a *different model* on the same hardware still fails verification. This establishes that TOPLOC isn't just passing because proofs are too loose.

Ideally: run a smaller Qwen model (e.g., Qwen3-30B-A3B) on the same hardware, enforce the same tokens, and verify that proofs diverge. If no smaller model is available, skip this phase — Phase 1 alone is valuable.

### Phase 3: Long-context error accumulation

Specifically measure how proof error statistics change as a function of token position. Use the document summarization and RAG samples (longest contexts).

- Slice proofs by token-position bucket: [0–1K], [1K–4K], [4K–16K], [16K–32K], [32K+]
- Report exponent mismatch and mantissa error per bucket
- Critical question: does error growth (Figure 2 from paper) stay moderate at 30K+ tokens, or does it blow up?
- MoE-specific: do longer sequences accumulate more expert-routing divergence?

### Phase 4: Pipeline-parallelism depth

The 8×H100 (TP=4, PP=2) setup splits the 94 layers across 2 pipeline stages. The layer boundary means activations are communicated between GPUs mid-forward-pass. Test whether this inter-GPU communication introduces additional numerical noise vs TP-only.

Compare: 4×H100 (TP=4) vs 8×H100 (TP=4, PP=2) error statistics — both are positive-control (same model), but PP=2 may show higher baseline noise.

## Metrics

Same as previous experiments, per proof:
- **exact%**: proofs with zero exponent mismatch and zero mantissa error
- **exp_mismatch**: mean exponent bit mismatches per proof
- **mant_err**: mean mantissa difference (for matching-exponent values)

Plus:
- **error vs. position**: exp_mismatch and mant_err binned by token index (buckets: 0–1K, 1K–4K, 4K–16K, 16K–32K, 32K+)
- **per-sample accept rate**: fraction of proofs per sample that pass thresholds → false-positive-at-sequence-level analysis
- **storage efficiency**: bytes/token at each (k, batch_size) operating point
- **hardware pair heatmap**: for each (generation hardware, validation hardware) pair, report the best-case and worst-case error statistics across the hyperparameter grid

## Execution Plan

4 runs total: 1 INT4, 3 FP8.

**Fraud scenario**: provider claims FP8 but actually served INT4. Both on new GPUs.
**Honest scenario**: provider ran FP8 on old hardware, verifier checks on new hardware.

```
Run 1: INT4 free generation (4×H100 or 4×B200, TP=4)
  → Generates 1100 responses from scratch
  → Collects top-512 hidden states per output token
  → These are the "fraudulent" outputs — claimed to be from FP8 but actually INT4

Run 2: FP8 free generation (4×A100, TP=4) — HONEST REFERENCE
  → Generates 1100 responses from scratch
  → Collects top-512 hidden states per output token
  → This is the honest provider running on older hardware

Run 3: FP8 enforced with INT4 tokens (4×H100 or 4×B200, TP=4) — NEGATIVE CONTROL
  → Feeds Run 1's token sequences into the FP8 model on new hardware
  → Collects top-512 hidden states per token
  → Compare these hidden states against Run 1's hidden states
  → Expected result: MISMATCH (different model produced the tokens)

Run 4: FP8 enforced with FP8 tokens (4×H100 or 4×B200, TP=4) — POSITIVE CONTROL
  → Feeds Run 2's token sequences into FP8 on new hardware
  → Collects top-512 hidden states per token
  → Compare these hidden states against Run 2's hidden states
  → Expected result: MATCH (same model, different hardware)
```

### Token volume estimate

| Subset | N | Avg input tokens | Avg output tokens | Total input | Total output |
|--------|--:|----------------:|-----------------:|------------:|-------------:|
| Existing short | 1000 | ~50 | ~261 | 50K | 261K |
| New long-output | 100 | ~100 | ~15K | 10K | 1.5M |
| **Total** | **1100** | | | **~60K** | **~1.76M** |

Total tokens per run: ~1.82M (input + output). The long-output subset dominates: 100 samples × ~15K output tokens = ~1.5M output tokens.

### Throughput assumptions

Qwen3-235B-A22B has 22B active parameters per token (MoE). On 4×H100 (TP=4):

- **Prefill** (processing known tokens, compute-bound): ~5,000–10,000 tok/s. Attention is O(n²), so long sequences (20K+) are slower per token.
- **Decode** (autoregressive generation, memory-bandwidth-bound): ~1,000–2,000 tok/s aggregate with vLLM continuous batching across the 500 samples.
- **Model load time**: ~5 min for a 235B model from disk to 4 GPUs.
- **TOPLOC overhead**: top-512 extraction + write to disk adds ~10–20% to forward pass time.

On 4×A100: ~60–70% of H100 throughput (lower HBM bandwidth: 2 TB/s vs 3.35 TB/s per GPU).
On 4×B200: likely ≥ H100 throughput.

### Time estimates per run

The bottleneck is the 100 long-output samples: 100 × ~15K output tokens = ~1.5M decode tokens. The 1000 short samples add ~261K decode tokens. Total decode per free-gen run: ~1.76M tokens.

| Run | Type | What happens | Hardware | Est. time |
|-----|------|-------------|----------|-----------|
| 1 | INT4 free gen | Prefill 60K input tok + decode 1.76M output tok | 4×H100 or 4×B200 | 20–35 min |
| 2 | FP8 free gen | Prefill 60K input tok + decode 1.76M output tok | 4×A100 | 30–50 min |
| 3 | FP8 enforced | Forward pass over 1.82M tokens (prefill-like, no sampling) | 4×H100 or 4×B200 | 5–10 min |
| 4 | FP8 enforced | Forward pass over 1.82M tokens (prefill-like, no sampling) | 4×H100 or 4×B200 | 5–10 min |

**Enforced generation is faster than free generation** because all tokens are known ahead of time — the model processes them in chunks (like prefill) rather than one-at-a-time autoregressive decode.

Run 2 (A100) is the slowest: older hardware (~60–70% of H100 throughput) doing free generation of the full 1.76M tokens.

| | Optimistic | Realistic (with overhead) |
|---|--------:|----------:|
| Pure compute (4 runs) | 1 hour | 1.75 hours |
| Model loading (4×, ~5 min each) | 20 min | 20 min |
| Data transfer between setups | — | 15 min |
| Debugging / restarts | — | 30–60 min |
| **Total wall-clock** | **~1.5 hours** | **~2.5–3 hours** |

Offline evaluation (building proofs and verifying across the hyperparameter grid) is cheap — pure NumPy/CPU, minutes at most.

## Open Questions

1. **MoE routing reproducibility**: Is expert gating deterministic across GPU architectures for the same input? If not, TOPLOC on MoE models may require higher k or different thresholds to absorb routing-induced noise. Phase 1 will answer this directly.
2. **Pipeline parallelism noise**: Does PP=2 introduce additional floating-point divergence at the pipeline stage boundary? The paper did not test PP at all.
3. **FP8 KV cache × 94 layers**: Our Qwen 3B experiments used FP8 KV cache with ~30 layers. With 94 layers, cumulative FP8 quantization error in the KV cache may be larger. Does this affect proof stability for long sequences?
4. **Threshold calibration**: The paper's thresholds (T_exp=38, T_mean=10, T_median=8) were calibrated on Llama 8B (dense). Phase 1 data will provide the baseline error distribution for this MoE model — thresholds will need recalibration.
5. **B200 FP8 behavior**: B200 (Blackwell) may use a different FP8 format or have different rounding behavior than H100 (Hopper). This is untested territory for TOPLOC.
