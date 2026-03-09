from datasets import load_dataset
from typing import List, Dict, Tuple


# Map user-facing language codes to HuggingFace dataset language codes
LANG_MAP = {
    "en": "en",
    "sp": "es",
    "ch": "zh",
    "hi": "hi",
    "ar": "ar",
}


def preload_all_language_prompts(
    langs: Tuple[str, ...] = ("en", "sp", "ch", "hi", "ar"),
) -> Dict[str, List[str]]:
    dataset = {}
    for lang in langs:
        hf_lang = LANG_MAP.get(lang, lang)
        data_files = f"hf://datasets/MBZUAI/Bactrian-X/data/{hf_lang}.json.gz"
        ds = load_dataset("json", data_files=data_files, split="train")
        prompts = [item["instruction"] for item in ds]
        dataset[lang] = prompts
        print(f"Loaded {len(prompts)} prompts for language '{lang}' (hf: '{hf_lang}')")
    return dataset


def slice_mixed_language_prompts_with_langs(
    dataset: Dict[str, List[str]],
    per_language_n: int,
    langs: Tuple[str, ...],
) -> Tuple[List[str], List[str]]:
    prompts = []
    languages = []
    for lang in langs:
        lang_prompts = dataset[lang][:per_language_n]
        prompts.extend(lang_prompts)
        languages.extend([lang] * len(lang_prompts))
    return prompts, languages
