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

    # ----- TOKENIZER LOADING -----
    if spec.tokenizer is not None:
        tokenizer_ref = spec.tokenizer
        use_subfolder = False
    else:
        tokenizer_ref = model_ref
        use_subfolder = True

    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "padding_side": "left",
        "use_fast": True,
    }
    if use_subfolder and spec.subfolder:
        tokenizer_kwargs["subfolder"] = spec.subfolder

    if is_local:
        tokenizer_kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
    except Exception as e:
        if is_local:
            tokenizer_kwargs.pop("local_files_only", None)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
        else:
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- MODEL LOADING -----
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

    if spec.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Handle config
    base_model_id = "Qwen/Qwen3.5-2B"
    try:
        config = AutoConfig.from_pretrained(model_ref, **model_kwargs)
        if not hasattr(config, "model_type") or config.model_type is None:
            base_config = AutoConfig.from_pretrained(
                base_model_id,
                cache_dir=cache_dir,
                trust_remote_code=spec.trust_remote_code,
            )
            for key, value in config.to_dict().items():
                if key != "model_type":
                    setattr(base_config, key, value)
            config = base_config
    except Exception as e:
        print(f"Warning: Could not load config from {model_ref}, falling back to {base_model_id}. Error: {e}")
        config = AutoConfig.from_pretrained(
            base_model_id,
            cache_dir=cache_dir,
            trust_remote_code=spec.trust_remote_code,
        )

    if spec.model_type and spec.model_type != "auto":
        config.model_type = spec.model_type

    model_kwargs["config"] = config

    # ----- LOAD MODEL (without strict parameter) -----
    # We'll handle strict loading manually after prefix stripping if needed
    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)

    # ----- HANDLE LANGUAGE_MODEL PREFIX STRIPPING -----
    if spec.strip_language_model_prefix:
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        print("Stripping 'model.language_model.' prefix from checkpoint...")
        
        # Determine the weight file (safetensors preferred)
        try:
            model_file = hf_hub_download(
                repo_id=model_ref,
                filename="model.safetensors",
                subfolder=spec.subfolder or "",
                cache_dir=cache_dir,
                local_files_only=is_local,
            )
            state_dict = safetensors.torch.load_file(model_file)
        except Exception:
            # Fallback to PyTorch bin
            try:
                model_file = hf_hub_download(
                    repo_id=model_ref,
                    filename="pytorch_model.bin",
                    subfolder=spec.subfolder or "",
                    cache_dir=cache_dir,
                    local_files_only=is_local,
                )
                state_dict = torch.load(model_file, map_location="cpu")
            except Exception as e:
                print(f"Warning: Could not load model weights for prefix stripping: {e}")
                print("Continuing with partially loaded model...")
                model.eval()
                return model, tokenizer

        # Remap keys: remove "model.language_model." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load the remapped state dict with strict=False to handle any remaining mismatches
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys after remap: {missing[:10] if len(missing) > 10 else missing}... (total: {len(missing)})")
        if unexpected:
            print(f"Unexpected keys after remap: {unexpected[:10] if len(unexpected) > 10 else unexpected}... (total: {len(unexpected)})")
        
        # Only raise if there are critical missing keys (not just tied weights)
        critical_missing = [k for k in missing if "tied" not in k and "weight" in k]
        if len(critical_missing) > 20:
            raise RuntimeError(f"Too many critical missing keys after remap: {len(critical_missing)}")

    model.eval()
    return model, tokenizer