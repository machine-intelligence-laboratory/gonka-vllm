"""Evaluate toploc verification on collected hidden states and logprobs.

Supports two data formats:
  - Full: NPZ with dense hidden_states (N, hidden_dim) and sparse logprobs
  - Reduced: NPZ with sparse topk_hidden_{indices,values} (N, 128) and sparse logprobs

Supports:
  - Self-verification (baseline): build and verify against same data
  - Perturbation test: add noise to test sensitivity
  - Cross-run comparison: verify between two sets of data

Usage:
    # Self-verification sweep
    python evaluate.py --npz-dir collected_logs/vllm_topk

    # Cross-run comparison (matched by prompt via JSONL)
    python evaluate.py \\
        --npz-dir collected_logs/run_a/vllm_topk \\
        --npz-dir-b collected_logs/run_b/vllm_topk \\
        --jsonl collected_logs/run_a/logs.jsonl \\
        --jsonl-b collected_logs/run_b/logs.jsonl

    # Single configuration
    python evaluate.py --npz-dir ... --mode logprobs --k 4 --decode-batch-size 1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from toploc import build_proofs, verify_proofs


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    exp_mismatches: int
    mant_err_mean: float
    mant_err_median: float

    @property
    def is_exact(self):
        return self.exp_mismatches == 0 and self.mant_err_mean == 0.0


@dataclass
class ExperimentResult:
    mode: str
    k: int
    decode_batch_size: int
    results: list[VerifyResult] = field(default_factory=list)
    skipped: int = 0

    @property
    def n_total(self):
        return len(self.results)

    @property
    def n_exact(self):
        return sum(1 for r in self.results if r.is_exact)

    @property
    def pct_exact(self):
        return 100.0 * self.n_exact / self.n_total if self.n_total else 0.0

    @property
    def mean_exp_mismatch(self):
        if not self.results:
            return 0.0
        return np.mean([r.exp_mismatches for r in self.results])

    @property
    def mean_mant_err(self):
        if not self.results:
            return 0.0
        return np.mean([r.mant_err_mean for r in self.results])


# ---------------------------------------------------------------------------
# Tensor construction — all tokens, no position selection
# ---------------------------------------------------------------------------

def make_hidden_state_tensor(hidden_states: np.ndarray) -> torch.Tensor:
    """hidden_states (N, hidden_dim) float16 → bfloat16 tensor."""
    return torch.from_numpy(hidden_states.astype(np.float32)).to(torch.bfloat16)


def make_sparse_hidden_state_tensor(
    indices: np.ndarray,
    values: np.ndarray,
    hidden_dim: int,
) -> torch.Tensor:
    """Reconstruct dense hidden states from sparse top-k representation.

    indices: (N, top_k_stored) int32 — dimension indices
    values:  (N, top_k_stored) float16 — values at those dims
    Returns: (N, hidden_dim) bfloat16
    """
    n_tokens = indices.shape[0]
    tensor = torch.zeros(n_tokens, hidden_dim, dtype=torch.bfloat16)
    for i in range(n_tokens):
        idx = indices[i]
        vals = values[i]
        tensor[i, idx] = torch.from_numpy(vals.astype(np.float32)).to(torch.bfloat16)
    return tensor


def make_logprob_tensor(
    token_ids: np.ndarray,
    values: np.ndarray,
    vocab_size: int,
) -> torch.Tensor:
    """Build dense probability vectors for all tokens.

    Uses exp(logprob) so the most probable tokens have the largest values,
    matching toploc's top-k-by-absolute-value selection.
    """
    n_tokens = token_ids.shape[0]
    tensor = torch.zeros(n_tokens, vocab_size, dtype=torch.bfloat16)
    for i in range(n_tokens):
        ids = token_ids[i]
        vals = values[i]
        probs = np.exp(vals.astype(np.float32))
        tensor[i, ids] = torch.from_numpy(probs).to(torch.bfloat16)
    return tensor


def is_reduced_format(data: dict) -> bool:
    """Check if NPZ data uses the reduced sparse format."""
    return "topk_hidden_indices" in data


def load_tensor(
    data: dict, mode: str, vocab_size: int, hidden_dim: int = 0,
) -> torch.Tensor:
    if mode == "hidden_states":
        if is_reduced_format(data):
            return make_sparse_hidden_state_tensor(
                data["topk_hidden_indices"],
                data["topk_hidden_values"],
                hidden_dim,
            )
        return make_hidden_state_tensor(data["hidden_states"])
    else:
        return make_logprob_tensor(
            data["topk_logprob_token_ids"],
            data["topk_logprob_values"],
            vocab_size,
        )


# ---------------------------------------------------------------------------
# Toploc helpers
# ---------------------------------------------------------------------------

def _build_and_verify(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    k: int,
    decode_batch_size: int,
) -> list[VerifyResult]:
    """Build proofs from tensor_a, verify against tensor_b.

    All tokens are decode tokens — skip_prefill=True so toploc batches
    them all via decode_batching_size without treating row 0 as prefill.
    """
    proofs = build_proofs(
        tensor_a, decode_batching_size=decode_batch_size,
        topk=k, skip_prefill=True,
    )
    vr_list = verify_proofs(
        tensor_b, proofs, decode_batching_size=decode_batch_size,
        topk=k, skip_prefill=True,
    )

    return [
        VerifyResult(
            exp_mismatches=vr.exp_mismatches,
            mant_err_mean=vr.mant_err_mean,
            mant_err_median=vr.mant_err_median,
        )
        for vr in vr_list
    ]


# ---------------------------------------------------------------------------
# Cross-directory matching
# ---------------------------------------------------------------------------

def match_files_by_prompt(
    npz_dir_a: str,
    jsonl_a: str,
    npz_dir_b: str,
    jsonl_b: str,
) -> list[tuple[str, str]]:
    """Match NPZ files between two directories via prompt text in JSONL."""

    def _load_mapping(path):
        m = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                m[item["response_id"]] = item["prompt"]
        return m

    map_a = _load_mapping(jsonl_a)
    map_b = _load_mapping(jsonl_b)

    prompt_to_b = {v: k for k, v in map_b.items()}

    pairs = []
    for rid_a, prompt in map_a.items():
        rid_b = prompt_to_b.get(prompt)
        if rid_b is None:
            continue
        pa = os.path.join(npz_dir_a, f"{rid_a}.npz")
        pb = os.path.join(npz_dir_b, f"{rid_b}.npz")
        if os.path.exists(pa) and os.path.exists(pb):
            pairs.append((pa, pb))

    return pairs


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def infer_vocab_size(npz_files: list[str], n_sample: int = 20) -> int:
    """Infer vocab size from a sample of NPZ files."""
    max_id = 0
    for f in npz_files[:n_sample]:
        data = np.load(f)
        if "topk_logprob_token_ids" in data:
            max_id = max(max_id, int(data["topk_logprob_token_ids"].max()))
    # Round up to next multiple of 1024
    return ((max_id + 1024) // 1024) * 1024


def infer_hidden_dim(npz_files: list[str], n_sample: int = 20) -> int:
    """Infer hidden dimension from sparse format or dense shape."""
    max_idx = 0
    for f in npz_files[:n_sample]:
        data = np.load(f)
        if "topk_hidden_indices" in data:
            max_idx = max(max_idx, int(data["topk_hidden_indices"].max()))
        elif "hidden_states" in data:
            return data["hidden_states"].shape[1]
    # Round up to next power of 2
    dim = max_idx + 1
    p = 1
    while p < dim:
        p *= 2
    return p


def run_experiment(
    pairs: list[tuple[str, str | None]],
    mode: str,
    k: int,
    decode_batch_size: int,
    vocab_size: int,
    hidden_dim: int = 0,
    perturbation: float = 0.0,
) -> ExperimentResult:
    """Run toploc build + verify for one (mode, k, decode_batch_size) config."""
    result = ExperimentResult(mode=mode, k=k, decode_batch_size=decode_batch_size)

    for npz_a, npz_b in pairs:
        data_a = dict(np.load(npz_a))

        tensor_a = load_tensor(data_a, mode, vocab_size, hidden_dim)

        # Verification target
        if npz_b is not None and npz_b != npz_a:
            data_b = dict(np.load(npz_b))
            tensor_b = load_tensor(data_b, mode, vocab_size, hidden_dim)
        elif perturbation > 0:
            noise = torch.randn_like(tensor_a.float()) * perturbation
            tensor_b = (tensor_a.float() + noise).to(torch.bfloat16)
        else:
            tensor_b = tensor_a

        vr_list = _build_and_verify(tensor_a, tensor_b, k, decode_batch_size)
        result.results.extend(vr_list)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate toploc verification across configurations."
    )
    parser.add_argument(
        "--npz-dir", required=True, help="Directory with NPZ files"
    )
    parser.add_argument(
        "--npz-dir-b",
        help="Second NPZ directory for cross-run verification",
    )
    parser.add_argument("--jsonl", help="JSONL from collect.py (for matching)")
    parser.add_argument("--jsonl-b", help="Second JSONL (for matching)")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["hidden_states", "logprobs", "both"],
    )
    parser.add_argument(
        "--k",
        default="64,128,256",
        help="Top-k values, comma-separated (default: 64,128,256)",
    )
    parser.add_argument(
        "--decode-batch-size",
        default="1,10,100",
        help="Decode batching sizes, comma-separated (default: 1,10,100)",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.0,
        help="Additive noise std-dev for perturbation test (default: 0 = self-verify)",
    )
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    # ── parse parameter grids ──
    k_values = [int(x) for x in args.k.split(",")]
    batch_sizes = [int(x) for x in args.decode_batch_size.split(",")]
    modes = (
        ["hidden_states", "logprobs"] if args.mode == "both" else [args.mode]
    )

    # ── load file pairs ──
    npz_files_a = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    print(f"Found {len(npz_files_a)} NPZ files in {args.npz_dir}")

    if args.npz_dir_b and args.jsonl and args.jsonl_b:
        matched = match_files_by_prompt(
            args.npz_dir, args.jsonl, args.npz_dir_b, args.jsonl_b
        )
        pairs: list[tuple[str, str | None]] = matched
        print(f"Matched {len(pairs)} file pairs across directories")
    else:
        pairs = [(f, None) for f in npz_files_a]

    # ── vocab size & hidden dim ──
    all_npz = npz_files_a
    vocab_size = infer_vocab_size(all_npz)
    hidden_dim = infer_hidden_dim(all_npz)
    print(f"Vocab size (inferred): {vocab_size}")
    print(f"Hidden dim (inferred): {hidden_dim}")

    label = "self-verify"
    if args.npz_dir_b:
        label = "cross-verify"
    elif args.perturbation > 0:
        label = f"perturbation={args.perturbation}"
    print(f"Experiment: {label}\n")

    # ── sweep ──
    all_results: list[ExperimentResult] = []
    total_configs = len(modes) * len(k_values) * len(batch_sizes)

    with tqdm(total=total_configs, desc="Configs") as pbar:
        for mode in modes:
            for k in k_values:
                for bs in batch_sizes:
                    pbar.set_postfix_str(
                        f"{mode[:6]} k={k} bs={bs}"
                    )
                    exp = run_experiment(
                        pairs,
                        mode,
                        k,
                        bs,
                        vocab_size,
                        hidden_dim=hidden_dim,
                        perturbation=args.perturbation,
                    )
                    all_results.append(exp)
                    pbar.update(1)

    # ── summary table ──
    hdr = (
        f"{'mode':<16} {'k':>4} {'bs':>5} "
        f"{'exact':>9} {'pct':>7} {'exp_mis':>8} {'mant_err':>10} "
        f"{'skip':>5}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for exp in all_results:
        print(
            f"{exp.mode:<16} {exp.k:>4} {exp.decode_batch_size:>5} "
            f"{exp.n_exact:>4}/{exp.n_total:<4} {exp.pct_exact:>6.1f}% "
            f"{exp.mean_exp_mismatch:>8.2f} {exp.mean_mant_err:>10.4f} "
            f"{exp.skipped:>5}"
        )
    print("=" * len(hdr))

    # ── optional JSON output ──
    if args.output:
        out = [
            {
                "mode": e.mode,
                "k": e.k,
                "decode_batch_size": e.decode_batch_size,
                "n_exact": e.n_exact,
                "n_total": e.n_total,
                "pct_exact": round(e.pct_exact, 2),
                "mean_exp_mismatch": round(float(e.mean_exp_mismatch), 4),
                "mean_mant_err": round(float(e.mean_mant_err), 4),
                "skipped": e.skipped,
                "per_file": [
                    {
                        "exp_mismatches": r.exp_mismatches,
                        "mant_err_mean": round(r.mant_err_mean, 4),
                        "mant_err_median": round(r.mant_err_median, 4),
                    }
                    for r in e.results
                ],
            }
            for e in all_results
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
