A clarification of the ticket description above, based on the offline
benchmarks we ran on Qwen3-235B-A22B-Instruct-2507.

## Summary

Replace the current top-k-logprobs similarity check with a TOPLOC-style
polynomial commitment over top-k hidden-state activations, exposed through
the same validation flow. TOPLOC makes artifact size tunable: at the
~40 B/token the current logprobs check already pays, it gives a comparable
detection rate; and it can win up to ~10× in artifact size while still
maintaining a good-enough detection rate.

## Motivation

The current logprobs-similarity check works well — on the fraud scenarios
we tested it gives comparable or even slightly better detection rates than
TOPLOC at any matching cost. The limitation is that it is hard to tune:
artifact size and detection quality are tied to a single mechanism (top-N
logprobs + a similarity heuristic), with no clean way to either shrink the
artifact or push detection up.

TOPLOC ([arXiv:2501.16007](https://arxiv.org/abs/2501.16007)) gives the
same task two explicit knobs:

- `k` (top-k of the hidden state) sets per-proof size.
- `decode_batch_size` (tokens per proof) sets how many tokens one proof
  covers.

Together they trace an explicit cost / detection-quality frontier that
the protocol can pick a point on per request class, instead of a fixed
operating point baked into the validation mechanism.

In our offline benchmark on Qwen3-235B-A22B-Instruct-2507 (1100 multilingual
prompts, FP8 weights, FP8 KV cache, A100 ↔ H100 cross-architecture
positives, INT4-served-as-FP8 fraud as negative), the frontier at
FPR ≤ 0.5% on honest cross-arch pairs is:

| operating point    |   k |  bs | bytes/token | FNR vs INT4 fraud         |
|--------------------|-----|-----|-------------|---------------------------|
| strict reference   | 512 |   8 |       129.6 | 0.0%                      |
| strict reference   | 512 |  16 |        65.7 | 0.0%                      |
| match-current size | 512 |  32 |        33.7 | <= 1% (0.0% at 1% FPR)    |
| 2x smaller         |  64 |   8 |        16.4 | 4.5%                      |
| 2x smaller         | 256 |  32 |        16.5 | 4.0%                      |
| 10x smaller        |  64 |  32 |         4.2 | ~15%                      |

Even at ~10× smaller artifacts, the false-negative rate stays low enough
to detect fraud with high confidence.

And because the size is a knob rather than a fixed cost, it can later be
made adaptive — varied per model type, prompt length, request importance /
risk class, or any other policy the network wants — giving the protocol a
flexibility the current fixed-shape logprobs check does not have.

On this dataset we can't directly compare TOPLOC's FPR against the
logprobs check. The few prompts that fail produce very different outputs
on A100 vs H100, so the model itself is behaving differently across
architectures and it's not a TOPLOC issue. We'd need more experiments to
pin this down. Details in the research summary.

## Impact

- `gonka-ai/vllm`:
  - Instrument the GPU model runner to compute and emit top-k hidden-state
    proofs alongside (or instead of) the existing logprobs path. A working
    branch (`mil/topk-collect`) already does the GPU-side capture and dumps
    `.npz` artifacts; what's missing is the proof construction and the
    `/v1/validate` integration.
  - Two integration shapes are plausible — see *Proposed approach* below.
- `gonka-ai/gonka` (mlnode + decentralized-api): proof construction and
  verification both live on the vLLM side. The gonka changes are limited
  to running the new endpoint, persisting the new proof artifact format
  alongside (or instead of) the current logprobs payload, and calling
  the updated `/v1/validate` shape.
- Tests: end-to-end honest + fraud scenarios on a small model
  (Qwen2.5-3B) and the production model (Qwen3-235B-A22B-Instruct-2507),
  including cross-architecture pairs.

## Expected outcome

- `/v1/validate` returns a TOPLOC verdict alongside (or replacing) the
  current similarity score. Hidden states are never exposed to the client.
- Default thresholds calibrated for the production model and at least the
  A100 ↔ H100 and H100 ↔ H100 hardware pairs.
- A documented size / detection-quality curve so the protocol can pick a
  point and revisit it as new hardware joins the network.

  
## Experimental results

Full report with figures: `toploc_reports/gonka_research_summary.md`.
