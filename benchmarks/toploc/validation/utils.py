import logging
import requests
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from validation.data import (
    ModelInfo,
    RequestParams,
    ExperimentRequest,
    ValidationItem,
    Result,
    PositionResult,
)


logger = logging.getLogger(__name__)


class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)


class EnforcedTokens(BaseModel):
    tokens: List[EnforcedToken]

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)

    @classmethod
    def from_result(cls, result: Result) -> "EnforcedTokens":
        return cls(
            tokens=[
                EnforcedToken(token=r.token, top_tokens=list(r.logprobs.keys()))
                for r in result.results
            ]
        )


def _prepare_messages(prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Response clear, correct and complete."},
        {"role": "user", "content": prompt},
    ]


def inference(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "n": 1,
        "top_logprobs": request_params.top_logprobs,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "return_tokens_as_token_ids": True,
    }
    for key, value in request_params.additional_params.items():
        payload[key] = value

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"Inference API request failed with status {response.status_code} {response.text}"
        )
    return response.json()


def validation(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
    enforced_str: Optional[str] = None,
    enforced_tokens: Optional[EnforcedTokens] = None,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": request_params.top_logprobs,
        "n": 1,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "return_tokens_as_token_ids": True,
    }
    for key, value in request_params.additional_params.items():
        payload[key] = value

    if enforced_str:
        payload["enforced_str"] = enforced_str
    if enforced_tokens:
        payload["enforced_tokens"] = enforced_tokens.dict()

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"Validation API request failed with status {response.status_code} {response.text}"
            f"\n(enforced_tokens: {enforced_tokens})\n(payload: {payload})"
        )
    return response.json()


def _extract_logprobs(resp) -> Result:
    logprobs = resp["choices"][0]["logprobs"]["content"]
    text = resp["choices"][0]["message"]["content"]
    results = []
    for position in logprobs:
        pos_result = PositionResult(
            token=position["token"],
            logprobs={lp["token"]: lp["logprob"] for lp in position["top_logprobs"]},
        )
        results.append(pos_result)

    return Result(text=text, results=results)


def _extract_enforced_tokens(resp) -> EnforcedTokens:
    return EnforcedTokens.from_content(resp["choices"][0]["logprobs"]["content"])


def generate_and_validate(
    experiment_request: ExperimentRequest,
) -> ValidationItem:
    inference_resp = inference(
        experiment_request.inference_model,
        experiment_request.request_params,
        experiment_request.prompt,
    )
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)

    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        experiment_request.prompt,
        enforced_tokens=enforced_tokens,
    )
    validation_result = _extract_logprobs(validation_resp)

    if validation_result.text != inference_result.text:
        logger.warning(
            "text sequences don't match\n"
            "inference:\n %s\n"
            "%s\n"
            "validation:\n %s\n"
            "%s",
            inference_result.text,
            "-" * 10,
            validation_result.text,
            "-" * 100,
        )

    return experiment_request.to_result(inference_result, validation_result)
