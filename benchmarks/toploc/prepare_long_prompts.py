"""Select 100 substantive Bactrian-X prompts and wrap them for long-form output.

Produces prompts expected to generate 10K-30K tokens when served with
max_new_tokens=32768. Uses the same Bactrian-X source as the existing
1000-prompt set, but filters for topics that naturally require long responses
and adds a system prompt encouraging comprehensive, detailed output.

Usage:
    python prepare_long_prompts.py --output prompts_long.json
    python prepare_long_prompts.py --output prompts_long.json --n-prompts 100
"""
import argparse
import json
import re

from validation.prompts import preload_all_language_prompts

# Keywords that signal a prompt likely to produce long output
LONG_KEYWORDS = re.compile(
    r"\b("
    r"explain|describe|discuss|write|create|design|develop|analyze|compare"
    r"|outline|summarize|provide|generate|compose|draft|elaborate"
    r"|how does|how do|how can|how to|what are|what is the"
    r"|advantages|disadvantages|differences|similarities|impact|effects"
    r"|history|process|steps|strategy|plan|guide|overview|review"
    r")\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a thorough, knowledgeable writer. Provide an extremely "
    "comprehensive and detailed response. Cover every major aspect of the "
    "topic with specific examples, explanations, and analysis. Structure "
    "your response with clear sections. Be extensive — aim for a complete, "
    "in-depth treatment that leaves no important angle unexplored."
)

# Skip prompts that are too vague or context-dependent
SKIP_PATTERNS = re.compile(
    r"(identify the odd one|following (fraction|sentence|paragraph|text|code)|"
    r"given (sentence|text|input|list|table)|rewrite the|edit the|"
    r"classify the|categorize the|fill in the blank|translate the|"
    r"convert the|fix the|correct the)",
    re.IGNORECASE,
)


def score_prompt(prompt: str) -> float:
    """Score a prompt by how likely it is to produce a long response."""
    if len(prompt) < 20:
        return 0.0
    if SKIP_PATTERNS.search(prompt):
        return 0.0
    matches = len(LONG_KEYWORDS.findall(prompt))
    length_bonus = min(len(prompt) / 200, 1.0)
    return matches + length_bonus


def main():
    parser = argparse.ArgumentParser(
        description="Prepare long-output prompts for TOPLOC stress testing."
    )
    parser.add_argument(
        "--output",
        default="prompts_long.json",
        help="Output path (default: prompts_long.json)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=100,
        help="Number of prompts to select (default: 100)",
    )
    args = parser.parse_args()

    print("Loading Bactrian-X English prompts...")
    dataset = preload_all_language_prompts(langs=("en",))
    all_prompts = dataset["en"]
    print(f"  {len(all_prompts)} total English prompts")

    scored = [(score_prompt(p), p) for p in all_prompts]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for score, prompt in scored:
        if score <= 0:
            continue
        # Deduplicate by first 50 chars
        key = prompt[:50].lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(prompt)
        if len(selected) >= args.n_prompts:
            break

    print(f"  Selected {len(selected)} prompts (min score threshold: >0)")

    data = [
        {
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "language": "en",
        }
        for prompt in selected
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} long-output prompts to {args.output}")
    print(f"\nSample prompts:")
    for entry in data[:5]:
        print(f"  - {entry['prompt'][:100]}...")


if __name__ == "__main__":
    main()
