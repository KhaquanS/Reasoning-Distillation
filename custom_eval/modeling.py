from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from custom_eval.config import ModelSpec


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[dtype_name]


def load_model_and_tokenizer(spec: ModelSpec, cache_dir: str | None):
    model_ref = str(Path(spec.checkpoint).expanduser()) if Path(spec.checkpoint).exists() else spec.checkpoint
    tokenizer_ref = spec.tokenizer or model_ref

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        cache_dir=cache_dir,
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": resolve_dtype(spec.dtype),
        "device_map": spec.device_map,
    }
    if spec.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.eval()
    return model, tokenizer

