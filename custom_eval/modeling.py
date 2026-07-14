"""
Model loading utilities with support for HF models, local checkpoints, and subfolders.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
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


def is_local_path(path: str) -> bool:
    """Check if the given path is a local directory containing config.json."""
    p = Path(path).expanduser()
    return p.exists() and p.is_dir() and (p / "config.json").exists()


def load_model_and_tokenizer(
    spec: ModelSpec,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from Hugging Face, a local path, or a subfolder inside a repo.
    """
    # Determine if checkpoint is a local folder or a Hub repo ID
    checkpoint_path = Path(spec.checkpoint).expanduser()
    if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
        model_ref = str(checkpoint_path)
        is_local = True
    else:
        model_ref = spec.checkpoint
        is_local = False

    tokenizer_ref = spec.tokenizer or model_ref

    # Build tokenizer kwargs
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "padding_side": "left",          # required for batched generation
    }
    if spec.subfolder:
        tokenizer_kwargs["subfolder"] = spec.subfolder

    if is_local:
        tokenizer_kwargs["local_files_only"] = True

    # Load tokenizer with retry fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
    except Exception as e:
        if is_local:
            # If local loading fails, try without local_files_only (might be a repo)
            tokenizer_kwargs.pop("local_files_only", None)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
        else:
            raise e

    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model loading kwargs
    model_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": resolve_dtype(spec.dtype),
        "device_map": spec.device_map,
    }
    if spec.subfolder:
        model_kwargs["subfolder"] = spec.subfolder

    # Quantization
    if spec.load_in_8bit and spec.load_in_4bit:
        raise ValueError("Cannot use both load_in_8bit and load_in_4bit")

    if spec.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=resolve_dtype(spec.dtype),
        )
    elif spec.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_dtype(spec.dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Flash Attention
    if spec.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # ----- FIX: Handle missing model_type in config -----
    # Try to load config from the checkpoint, but if it fails or lacks model_type,
    # fall back to the base Qwen config.
    base_model_id = "Qwen/Qwen3.5-2B"  # base model for architecture
    try:
        # Try loading config from the checkpoint (with subfolder if any)
        config = AutoConfig.from_pretrained(model_ref, **model_kwargs)
        # Check if model_type is present; if not, set it
        if not hasattr(config, "model_type") or config.model_type is None:
            # Load base config and copy over the attributes we need
            base_config = AutoConfig.from_pretrained(
                base_model_id,
                cache_dir=cache_dir,
                trust_remote_code=spec.trust_remote_code,
            )
            # Override with our checkpoint's config but keep the model_type
            # We'll merge: use base config and then update with our config's attributes
            for key, value in config.to_dict().items():
                if key != "model_type":
                    setattr(base_config, key, value)
            config = base_config
    except Exception as e:
        # If loading config from checkpoint fails entirely, use base config
        print(f"Warning: Could not load config from {model_ref}, falling back to {base_model_id}. Error: {e}")
        config = AutoConfig.from_pretrained(
            base_model_id,
            cache_dir=cache_dir,
            trust_remote_code=spec.trust_remote_code,
        )
        # Also set any subfolder? The base config doesn't have subfolder; we rely on model_kwargs for that.

    # Pass the config to model loading
    model_kwargs["config"] = config

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    model.eval()

    return model, tokenizer