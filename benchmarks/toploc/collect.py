"""Collect tokens and hidden states from a running vLLM server.

Hidden states and top-k logprobs are collected server-side into .npz files
(controlled by the TOPLOC code in gpu_model_runner.py).  This script drives
the generation and saves a lightweight JSONL mapping each request to its
prompt and generated tokens.

Workflow for comparing two GPUs:
    1. Free run on GPU A  – generates tokens freely:
       python collect.py --server-url http://gpuA:8000 ...

    2. Enforced run on GPU B – replays the same tokens:
       python collect.py --server-url http://gpuB:8000 \
           --from-collection <output from step 1> ...

    The server-side .npz files already contain hidden states + top-512
    logprobs, so this script only stores {prompt, tokens, response_id}.

Usage:
    # On the server:
    VLLM_TOPLOC_OUTPUT_DIR=/data/toploc_hs vllm serve ...


"""
import argparse
import json
import logging
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from validation.utils import inference, validation, EnforcedTokens, EnforcedToken
from validation.data import ModelInfo, RequestParams

logger = logging.getLogger(__name__)


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_items_jsonl(items, path, append=False):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_items_jsonl(path, n=None):
    limit = n if n is not None else float("inf")
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            items.append(json.loads(line))
    return items


def _extract_tokens(resp):
    """Return the list of token strings from an API response."""
    content = resp["choices"][0]["logprobs"]["content"]
    return [pos["token"] for pos in content]


def collect_one(prompt_entry, model_info, request_params):
    """Free generation: send a prompt, get back tokens."""
    resp = inference(model_info, request_params, prompt_entry["prompt"])
    return {
        "prompt": prompt_entry["prompt"],
        "language": prompt_entry.get("language"),
        "tokens": _extract_tokens(resp),
        "response_id": resp.get("id"),
    }


def collect_one_enforced(prior_entry, model_info, request_params):
    """Enforced generation: replay tokens from a prior run."""
    enforced_tokens = EnforcedTokens(
        tokens=[EnforcedToken(token=t) for t in prior_entry["tokens"]]
    )
    resp = validation(
        model_info, request_params, prior_entry["prompt"],
        enforced_tokens=enforced_tokens,
    )
    return {
        "prompt": prior_entry["prompt"],
        "language": prior_entry.get("language"),
        "tokens": _extract_tokens(resp),
        "response_id": resp.get("id"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect toploc data from a running vLLM server."
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="vLLM server URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--model",
        help="Model name as served by vLLM",
        default="/data/shared/CompressaAI/test_deploy/models/models/"
                "Qwen_Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--prompts",
        help="Path to prompts JSON file from prepare_prompts.py",
        default="prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save results",
        default="/home/zenovkin_n/vllm_logs",
    )
    parser.add_argument("--gpu", default="V100", help="GPU label, e.g. 1xH100")
    parser.add_argument(
        "--precision", default="fp8", help="Precision label, e.g. fp8"
    )
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Collect only the first N prompts (default: all)",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--from-collection",
        default=None,
        help="Path to a previous collection JSONL. When set, sends enforced "
             "tokens from that run instead of generating freely.",
    )
    args = parser.parse_args()

    # ── Load prior run (enforced mode) or prompts (free mode) ──
    prior_items = None
    if args.from_collection:
        prior_items = load_items_jsonl(
            args.from_collection, n=args.num_samples
        )
        logger.info(
            "Loaded %d items from %s for enforced collection",
            len(prior_items), args.from_collection,
        )

    prompts = None
    if prior_items is None:
        prompts = load_prompts(args.prompts)
        if args.num_samples is not None:
            prompts = prompts[: args.num_samples]

    model_info = ModelInfo(
        url=args.server_url,
        name=args.model,
        deploy_params={"GPU": args.gpu, "precision": args.precision},
    )

    request_params = RequestParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        top_logprobs=1,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model_short = args.model.split("/")[-1]
    mode_tag = "enforced" if prior_items else "free"
    filename = f"{model_short}_{args.precision}_{args.gpu}_{mode_tag}.jsonl"
    output_path = os.path.join(args.output_dir, filename)

    n_items = len(prior_items) if prior_items else len(prompts)

    # Save config alongside results
    config_path = os.path.join(
        args.output_dir,
        f"{model_short}_{args.precision}_{args.gpu}_{mode_tag}_config.json",
    )
    config = {
        "model": args.model,
        "server_url": args.server_url,
        "gpu": args.gpu,
        "precision": args.precision,
        "mode": mode_tag,
        "request_params": request_params.model_dump(),
        "n_prompts": n_items,
        "prompts_file": args.prompts,
        "from_collection": args.from_collection,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        if prior_items:
            futures = {
                executor.submit(
                    collect_one_enforced, item, model_info, request_params
                ): item
                for item in prior_items
            }
        else:
            futures = {
                executor.submit(collect_one, p, model_info, request_params): p
                for p in prompts
            }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Collecting"
        ):
            try:
                item = future.result()
                results.append(item)
                save_items_jsonl([item], output_path, append=True)
            except Exception:
                src = futures[future]
                if isinstance(src, dict):
                    prompt = src.get("prompt", "")
                else:
                    prompt = str(src)
                logger.exception("Failed for prompt: %s", prompt[:100])

    print(f"Done. {len(results)} results saved to {output_path}")
    print(
        "Hidden states + top-k logprobs are on the server in "
        "$VLLM_TOPLOC_OUTPUT_DIR. Join with this JSONL via 'response_id'."
    )


if __name__ == "__main__":
    main()
