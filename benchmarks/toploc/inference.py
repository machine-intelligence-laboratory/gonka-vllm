import sys
sys.path.append('validation')
sys.path.insert(0, '.')

import os
import copy
import json
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from validation.runner import run_validation
from validation.prompts import preload_all_language_prompts, slice_mixed_language_prompts_with_langs
from validation.data import ModelInfo, RequestParams, ServerConfig, RunParams, InferenceValidationRun
from validation.model_presets import LLAMA_31_8B_FP8, LLAMA_31_8B_INT4


N_PROMPTS = 1000
MAX_WORKERS = None

run_params_base = RunParams(
    exp_name='toploc_llama31_8b',
    output_path='./data/toploc_results',
    n_prompts=N_PROMPTS,
    timeout=1800,
    tokenizer_model_name='meta-llama/Llama-3.1-8B-Instruct',
    request=RequestParams(
        max_tokens=3000,
        temperature=0.99,
        seed=42,
        top_logprobs=5,
        additional_params={
            "toploc_k": 512,
            "toploc_logprobs_k": 512,
        },
    ),
)


def get_run_params(temp, prompts):
    params = copy.deepcopy(run_params_base)
    params.request.temperature = temp
    params.n_prompts = prompts
    return params


# Hardware configs — update IPs/ports for your servers
hardware_A = ServerConfig(
    ip='FILL_IN',
    inference_port='FILL_IN',
    node_port='FILL_IN',
    gpu='1xH100',
)

hardware_B = ServerConfig(
    ip='FILL_IN',
    inference_port='FILL_IN',
    node_port='FILL_IN',
    gpu='1xA6000',
)

langs = ("en", "sp", "ch", "hi", "ar")

runs = [
    # Honest FP8 on Hardware A vs FP8 on Hardware B
    InferenceValidationRun(
        model_inference=LLAMA_31_8B_FP8,
        model_validation=LLAMA_31_8B_FP8,
        server_inference=hardware_A,
        server_validation=hardware_B,
        run_inference=get_run_params(0.99, N_PROMPTS),
        run_validation=get_run_params(0.99, N_PROMPTS),
        max_workers=MAX_WORKERS,
    ),
    # Honest FP8 on Hardware B vs FP8 on Hardware A
    InferenceValidationRun(
        model_inference=LLAMA_31_8B_FP8,
        model_validation=LLAMA_31_8B_FP8,
        server_inference=hardware_B,
        server_validation=hardware_A,
        run_inference=get_run_params(0.99, N_PROMPTS),
        run_validation=get_run_params(0.99, N_PROMPTS),
        max_workers=MAX_WORKERS,
    ),
    # Fraudulent INT4 on Hardware A vs FP8 on Hardware B
    InferenceValidationRun(
        model_inference=LLAMA_31_8B_INT4,
        model_validation=LLAMA_31_8B_FP8,
        server_inference=hardware_A,
        server_validation=hardware_B,
        run_inference=get_run_params(0.99, N_PROMPTS),
        run_validation=get_run_params(0.99, N_PROMPTS),
        max_workers=MAX_WORKERS,
    ),
    # Fraudulent INT4 on Hardware B vs FP8 on Hardware A
    InferenceValidationRun(
        model_inference=LLAMA_31_8B_INT4,
        model_validation=LLAMA_31_8B_FP8,
        server_inference=hardware_B,
        server_validation=hardware_A,
        run_inference=get_run_params(0.99, N_PROMPTS),
        run_validation=get_run_params(0.99, N_PROMPTS),
        max_workers=MAX_WORKERS,
    ),
]


def post_up(server: ServerConfig, payload, timeout: int):
    response = requests.post(server.up_url(), json=payload, timeout=timeout)
    response.raise_for_status()
    return server, response


def main():
    dataset = preload_all_language_prompts(langs=langs)

    for cfg in runs:
        tokenizer_model_name = cfg.run_inference.tokenizer_model_name
        _ = AutoTokenizer.from_pretrained(tokenizer_model_name)

        os.makedirs(cfg.run_inference.output_path, exist_ok=True)

        inference_payload = cfg.model_inference.to_deploy_payload()
        validation_payload = cfg.model_validation.to_deploy_payload()
        up_requests = [
            (cfg.server_inference, inference_payload),
            (cfg.server_validation, validation_payload),
        ]
        print(up_requests)
        with ThreadPoolExecutor(max_workers=len(up_requests)) as executor:
            futures = [
                executor.submit(post_up, srv, payload, cfg.run_inference.timeout)
                for srv, payload in up_requests
            ]
            for future in futures:
                _, response = future.result()
                print(response.status_code)
                print(response.text)

        sleep(5)

        inference_model_info = ModelInfo(
            url=cfg.server_inference.inference_url(),
            name=cfg.model_inference.model,
            deploy_params={
                "GPU": cfg.server_inference.gpu,
                "precision": cfg.model_inference.precision,
            },
        )

        validation_model_info = ModelInfo(
            url=cfg.server_validation.inference_url(),
            name=cfg.model_validation.model,
            deploy_params={
                "GPU": cfg.server_validation.gpu,
                "precision": cfg.model_validation.precision,
            },
        )

        request_params = cfg.run_inference.request
        setting_name = cfg.setting_filename()
        data_path = f"{cfg.run_inference.output_path}/{setting_name}"
        n_prompts = cfg.run_inference.n_prompts

        # Save run configuration next to results
        config_filename = f"{setting_name.rsplit('.', 1)[0]}_config.json"
        config_path = f"{cfg.run_inference.output_path}/{config_filename}"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)

        prompts, languages = slice_mixed_language_prompts_with_langs(
            dataset, per_language_n=n_prompts // len(langs), langs=langs
        )
        _ = run_validation(
            prompts,
            languages=languages,
            inference_model=inference_model_info,
            validation_model=validation_model_info,
            request_params=request_params,
            max_workers=cfg.max_workers,
            output_path=data_path,
        )
        print(f"Completed run. Results saved to: {data_path}")


if __name__ == "__main__":
    main()
