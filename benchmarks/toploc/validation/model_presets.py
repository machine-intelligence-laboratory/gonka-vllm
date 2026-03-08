from validation.data import ModelPreset


LLAMA_31_8B_FP8 = ModelPreset(
    model='RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8',
    precision='fp8',
    dtype='float16',
    additional_args=[
        '--enforce-eager',
        '--gpu-memory-utilization', '0.95',
        '--max-model-len', '8192',
    ],
)

LLAMA_31_8B_INT4 = ModelPreset(
    model='hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
    precision='int4',
    dtype='float16',
    additional_args=[
        '--enforce-eager',
        '--gpu-memory-utilization', '0.95',
        '--max-model-len', '8192',
        '--quantization', 'awq',
    ],
)

QWEN25_3B = ModelPreset(
    model='Qwen/Qwen2.5-3B-Instruct',
    precision='fp16',
    dtype='float16',
    additional_args=[
        '--enforce-eager',
        '--gpu-memory-utilization', '0.95',
        '--max-model-len', '8192',
    ],
)
