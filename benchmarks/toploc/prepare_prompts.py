"""Save a fixed prompt set for reproducible toploc experiments.

Run once, then reuse the same prompts file across all collection runs.

Usage:
    python prepare_prompts.py --output prompts.json
    python prepare_prompts.py --output prompts.json --n-prompts 500 --langs en sp ch
"""
import argparse
import json

from validation.prompts import (
    preload_all_language_prompts,
    slice_mixed_language_prompts_with_langs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download and save a fixed prompt set for toploc experiments."
    )
    parser.add_argument(
        "--output", default="prompts.json", help="Output path for the prompts file"
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=1000,
        help="Total number of prompts (split evenly across languages)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "sp", "ch", "hi", "ar"],
        help="Language codes to include",
    )
    args = parser.parse_args()

    dataset = preload_all_language_prompts(langs=tuple(args.langs))
    per_lang = args.n_prompts // len(args.langs)
    prompts, languages = slice_mixed_language_prompts_with_langs(
        dataset, per_lang, tuple(args.langs)
    )

    data = [{"prompt": p, "language": l} for p, l in zip(prompts, languages)]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} prompts to {args.output}")


if __name__ == "__main__":
    main()