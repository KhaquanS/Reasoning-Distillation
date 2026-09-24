from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


SUPPORTED_BENCHMARKS = {
    "arc-c",
    "math500",
    "aime25",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "gpqa",
}


@dataclass
class ModelSpec:
    name: str
    checkpoint: str
    tokenizer: Optional[str] = None
    subfolder: Optional[str] = None
    trust_remote_code: bool = True
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    enable_thinking: bool = True
    model_type: str = "qwen"
    strip_language_model_prefix: bool = False   # <-- NEW: remap keys from model.language_model.* to model.*


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    benchmarks: list[str]
    output_dir: str = "./eval_outputs"
    cache_dir: Optional[str] = "./cache"
    split: str = "test"
    max_samples: Optional[int] = None
    seed: int = 42
    # Generation parameters
    max_new_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    pass_at_k: int | list[int] = 1
    num_samples: Optional[int] = None
    sample_batch_size: Optional[int] = None
    require_real_data: bool = False
    batch_size: int = 1
    benchmark_options: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path) -> EvalConfig:
    """Load evaluation configuration from YAML file."""
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    # Parse models
    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Config must include a non-empty models list.")

    models = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping.")

        checkpoint = item.get("checkpoint") or item.get("path") or item.get("model_id")
        if not checkpoint:
            raise ValueError("Each model needs checkpoint, path, or model_id.")

        # Model name
        model_name = item.get("name") or Path(str(checkpoint)).name

        # Determine default thinking mode based on model size if not explicitly set
        enable_thinking = item.get("enable_thinking")
        if enable_thinking is None:
            enable_thinking = "4b" in str(checkpoint).lower() or "4B" in str(checkpoint).lower()

        # Parse model_type, default to "qwen" for backward compatibility
        model_type = item.get("model_type", "qwen")

        models.append(
            ModelSpec(
                name=model_name,
                checkpoint=str(checkpoint),
                tokenizer=item.get("tokenizer"),
                subfolder=item.get("subfolder"),
                trust_remote_code=bool(item.get("trust_remote_code", True)),
                dtype=str(item.get("dtype", "auto")),
                device_map=str(item.get("device_map", "auto")),
                load_in_8bit=bool(item.get("load_in_8bit", False)),
                load_in_4bit=bool(item.get("load_in_4bit", False)),
                use_flash_attention_2=bool(item.get("use_flash_attention_2", False)),
                enable_thinking=enable_thinking,
                model_type=model_type,
                strip_language_model_prefix=bool(item.get("strip_language_model_prefix", False)),
            )
        )

    # Parse benchmarks
    aliases = {"aime-25": "aime25", "math-500": "math500"}
    benchmarks = [aliases.get(str(name).lower(), str(name).lower()) for name in raw.get("benchmarks", [])]
    unknown = sorted(set(benchmarks) - SUPPORTED_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not benchmarks:
        raise ValueError(f"At least one benchmark is required. Options: {sorted(SUPPORTED_BENCHMARKS)}")

    # Parse generation parameters
    gen = raw.get("generation", {}) or {}
    if not isinstance(gen, dict):
        raise ValueError("generation must be a mapping.")

    pass_at_k = gen.get("pass_at_k", 1)
    values = pass_at_k if isinstance(pass_at_k, list) else [pass_at_k]
    if not values or any(type(k) is not int or k <= 0 for k in values):
        raise ValueError("generation.pass_at_k must be a positive integer or a list of them.")
    if len(set(values)) != len(values):
        raise ValueError("generation.pass_at_k must not contain duplicates.")
    num_samples = gen.get("num_samples", max(values))
    if type(num_samples) is not int or num_samples < max(values):
        raise ValueError("generation.num_samples must be an integer >= every pass_at_k.")
    sample_batch_size = gen.get("sample_batch_size", num_samples)
    if type(sample_batch_size) is not int or not 1 <= sample_batch_size <= num_samples:
        raise ValueError("generation.sample_batch_size must be between 1 and num_samples.")
    if num_samples > 1 and float(gen.get("temperature", 1.0)) <= 0:
        raise ValueError("Multiple samples require temperature > 0 for stochastic sampling.")
    if int(gen.get("batch_size", 1)) <= 0:
        raise ValueError("generation.batch_size must be positive.")
    if raw.get("max_samples") is not None and int(raw["max_samples"]) <= 0:
        raise ValueError("max_samples must be positive or null.")

    return EvalConfig(
        models=models,
        benchmarks=benchmarks,
        output_dir=str(raw.get("output_dir", "./eval_outputs")),
        cache_dir=raw.get("cache_dir", "./cache"),
        split=str(raw.get("split", "test")),
        max_samples=raw.get("max_samples"),
        seed=int(raw.get("seed", 42)),
        max_new_tokens=int(gen.get("max_new_tokens", 32768)),
        temperature=float(gen.get("temperature", 1.0)),
        top_p=float(gen.get("top_p", 0.95)),
        top_k=int(gen.get("top_k", 20)),
        presence_penalty=float(gen.get("presence_penalty", 1.5)),
        repetition_penalty=float(gen.get("repetition_penalty", 1.0)),
        pass_at_k=pass_at_k,
        num_samples=num_samples,
        sample_batch_size=sample_batch_size,
        require_real_data=bool(raw.get("require_real_data", False)),
        batch_size=int(gen.get("batch_size", 1)),
        benchmark_options=raw.get("benchmark_options", {}) or {},
    )