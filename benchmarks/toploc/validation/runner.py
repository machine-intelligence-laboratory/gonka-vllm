import logging
from typing import List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from validation.utils import generate_and_validate
from validation.data import (
    ValidationItem,
    ModelInfo,
    RequestParams,
    ExperimentRequest,
    save_to_jsonl,
)
from tqdm import tqdm


logger = logging.getLogger(__name__)


def run_validation(
    prompts: List[str],
    languages: List[str],
    inference_model: ModelInfo,
    validation_model: ModelInfo,
    request_params: RequestParams,
    max_workers: Optional[int] = None,
    output_path: Optional[str] = None,
) -> List[ValidationItem]:
    args = [
        ExperimentRequest(
            prompt=prompt,
            language=language,
            inference_model=inference_model,
            validation_model=validation_model,
            request_params=request_params,
        )
        for prompt, language in zip(prompts, languages)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_and_validate, arg): arg for arg in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running validation"):
            try:
                result = future.result()
                results.append(result)
                if output_path:
                    save_to_jsonl([result], output_path, append=True)
            except Exception:
                prompt = futures[future].prompt
                logger.exception("Failed for prompt: %s", prompt[:100])

    return results
