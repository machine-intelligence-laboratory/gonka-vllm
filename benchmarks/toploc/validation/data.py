from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional


class PositionResult(BaseModel):
    token: str
    logprobs: Dict[str, float]
    toploc_indices: Optional[List[int]] = None
    toploc_values: Optional[List[float]] = None
    toploc_logprob_token_ids: Optional[List[int]] = None
    toploc_logprob_values: Optional[List[float]] = None


class Result(BaseModel):
    text: str
    results: List[PositionResult]


class ModelInfo(BaseModel):
    name: str
    url: str
    deploy_params: Dict[str, str] = Field(default_factory=dict)


class RequestParams(BaseModel):
    max_tokens: int
    temperature: float
    seed: int
    additional_params: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    top_logprobs: int = 3


class ValidationItem(BaseModel):
    prompt: str
    language: Optional[str] = None
    inference_result: Result
    validation_result: Result
    inference_model: ModelInfo
    validation_model: ModelInfo
    request_params: RequestParams

    def to_dict(self):
        return self.model_dump()


class ExperimentRequest(BaseModel):
    prompt: str
    language: Optional[str] = None
    inference_model: ModelInfo
    validation_model: ModelInfo
    request_params: RequestParams

    def to_result(self, inference_result: Result, validation_result: Result) -> ValidationItem:
        return ValidationItem(
            prompt=self.prompt,
            language=self.language,
            inference_result=inference_result,
            validation_result=validation_result,
            inference_model=self.inference_model,
            validation_model=self.validation_model,
            request_params=self.request_params,
        )


class ServerConfig(BaseModel):
    ip: str
    inference_port: str
    node_port: str
    gpu: str

    def up_url(self) -> str:
        return f"http://{self.ip}:{self.node_port}/up"

    def inference_url(self) -> str:
        return f"http://{self.ip}:{self.inference_port}"


class ModelPreset(BaseModel):
    model: str
    precision: str
    dtype: str
    additional_args: List[str] = Field(default_factory=list)

    def to_deploy_payload(self) -> dict:
        return {
            "model": self.model,
            "dtype": self.dtype,
            "additional_args": self.additional_args,
        }


class RunParams(BaseModel):
    exp_name: str
    output_path: str
    n_prompts: int
    timeout: int
    tokenizer_model_name: str
    request: RequestParams


class InferenceValidationRun(BaseModel):
    model_inference: ModelPreset
    model_validation: ModelPreset
    server_inference: ServerConfig
    server_validation: ServerConfig
    run_inference: RunParams
    run_validation: RunParams
    max_workers: Optional[int] = None

    def setting_filename(self) -> str:
        inf_model = self.model_inference.model.split('/')[-1]
        val_model = self.model_validation.model.split('/')[-1]
        inf_gpu = self.server_inference.gpu
        val_gpu = self.server_validation.gpu
        inf_prec = self.model_inference.precision
        val_prec = self.model_validation.precision
        return f"{inf_model}_{inf_prec}_{inf_gpu}___{val_model}_{val_prec}_{val_gpu}.jsonl"


def save_to_jsonl(
    validation_results: List[ValidationItem],
    path: str,
    append: bool = False,
):
    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        for result in validation_results:
            f.write(result.model_dump_json() + '\n')


def load_from_jsonl(
    path: str,
    n: Optional[int] = None,
) -> List[ValidationItem]:
    k = n if n is not None else float('inf')
    results = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            results.append(ValidationItem.model_validate_json(line))
    return results
