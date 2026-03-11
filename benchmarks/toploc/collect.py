"""Collect logprobs from a running vLLM server.

Hidden states are collected server-side via VLLM_TOPLOC_OUTPUT_DIR env var.
Start vLLM with that env var set, then point this script at it.

Each API response includes an ``id`` field (e.g. ``chatcmpl-<uuid>``).
The server writes ``<id>.npz`` files containing the raw hidden-state
vectors for every generated token.  This script saves a JSONL mapping
``response_id → prompt + logprobs`` so you can join them offline.

Usage:
    # On the server:
    VLLM_TOPLOC_OUTPUT_DIR=/data/toploc_hs vllm serve ...

    # On the client:
    python collect.py \\
        --server-url http://localhost:8000 \\
        --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \\
        --prompts prompts.json \\
        --output-dir ./data/run1 \\
        --gpu 1xH100 \\
        --precision fp8
"""
import argparse
import json
import logging
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from validation.utils import inference, _extract_logprobs
from validation.data import (
    CollectionItem,
    ModelInfo,
    RequestParams,
    save_collection_to_jsonl,
)

logger = logging.getLogger(__name__)


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_one(prompt_entry, model_info, request_params):
    resp = inference(model_info, request_params, prompt_entry["prompt"])
    result = _extract_logprobs(resp)
    response_id = resp.get("id")
    return CollectionItem(
        prompt=prompt_entry["prompt"],
        language=prompt_entry.get("language"),
        result=result,
        model=model_info,
        request_params=request_params,
        response_id=response_id,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect toploc data from a running vLLM server."
    )
    parser.add_argument(
        "--server-url",
        required=True, default="http://localhost:8000",
        help="vLLM server URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--model", required=True, help="Model name as served by vLLM", default="qwen",
    )
    parser.add_argument(
        "--prompts", required=True, help="Path to prompts JSON file from prepare_prompts.py", default="prompts.json"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save results", default="/home/zenovkin_n/vllm_logs"
    )
    parser.add_argument("--gpu", default="V100", help="GPU label, e.g. 1xH100")
    parser.add_argument(
        "--precision", default="fp8", help="Precision label, e.g. fp8"
    )
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-logprobs", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)

    model_info = ModelInfo(
        url=args.server_url,
        name=args.model,
        deploy_params={"GPU": args.gpu, "precision": args.precision},
    )

    request_params = RequestParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        top_logprobs=args.top_logprobs,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    model_short = args.model.split("/")[-1]
    filename = f"{model_short}_{args.precision}_{args.gpu}.jsonl"
    output_path = os.path.join(args.output_dir, filename)

    # Save config alongside results
    config_path = os.path.join(
        args.output_dir, f"{model_short}_{args.precision}_{args.gpu}_config.json"
    )
    config = {
        "model": args.model,
        "server_url": args.server_url,
        "gpu": args.gpu,
        "precision": args.precision,
        "request_params": request_params.model_dump(),
        "n_prompts": len(prompts),
        "prompts_file": args.prompts,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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
                save_collection_to_jsonl([item], output_path, append=True)
            except Exception:
                prompt = futures[future]["prompt"]
                logger.exception("Failed for prompt: %s", prompt[:100])

    print(f"Done. {len(results)} results saved to {output_path}")
    print(
        "Hidden states are on the server in $VLLM_TOPLOC_OUTPUT_DIR. "
        "Join with this JSONL via the 'response_id' field."
    )


if __name__ == "__main__":
    main()