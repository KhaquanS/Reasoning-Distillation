"""
Model loading utilities with support for HF models and local checkpoints.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from custom_eval.config import ModelSpec


def resolve_dtype(dtype_name: str) -> Union[str, torch.dtype]:
    """Convert dtype string to torch dtype."""
    if dtype_name == "auto":
        return "auto"
    
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Supported: {list(mapping.keys())}")
    
    return mapping[dtype_name]


def load_model_and_tokenizer(
    spec: ModelSpec,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from Hugging Face or local path.
    
    Args:
        spec: Model specification
        cache_dir: Cache directory for downloaded models
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine if checkpoint is a local path or HF model ID
    checkpoint_path = Path(spec.checkpoint).expanduser()
    if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
        model_ref = str(checkpoint_path)
    else:
        model_ref = spec.checkpoint
    
    tokenizer_ref = spec.tokenizer or model_ref
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        cache_dir=cache_dir,
        trust_remote_code=spec.trust_remote_code,
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build model loading kwargs
    kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": resolve_dtype(spec.dtype),
        "device_map": spec.device_map,
    }
    
    # Quantization
    if spec.load_in_8bit and spec.load_in_4bit:
        raise ValueError("Cannot use both load_in_8bit and load_in_4bit")
    
    if spec.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=resolve_dtype(spec.dtype),
        )
    elif spec.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_dtype(spec.dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Flash Attention
    if spec.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.eval()
    
    return model, tokenizer