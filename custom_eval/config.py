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
    trust_remote_code: bool = True
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    # Qwen-specific
    enable_thinking: bool = True  # For Qwen3.5-4B, True by default; for Qwen3.5-2B, False by default


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    benchmarks: list[str]
    output_dir: str = "./eval_outputs"
    cache_dir: Optional[str] = "./cache"
    split: str = "test"
    max_samples: Optional[int] = None
    seed: int = 42
    # Generation parameters (Qwen best practices)
    max_new_tokens: int = 32768  # Qwen recommends 32,768 for most queries
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    batch_size: int = 1
    # Benchmark-specific options
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

        # Qwen-specific: determine default thinking mode based on model size
        model_name = item.get("name") or Path(str(checkpoint)).name
        enable_thinking = item.get("enable_thinking")
        if enable_thinking is None:
            # Qwen3.5-4B defaults to thinking mode, Qwen3.5-2B defaults to non-thinking
            enable_thinking = "4b" in str(checkpoint).lower() or "4B" in str(checkpoint).lower()

        models.append(
            ModelSpec(
                name=model_name,
                checkpoint=str(checkpoint),
                tokenizer=item.get("tokenizer"),
                trust_remote_code=bool(item.get("trust_remote_code", True)),
                dtype=str(item.get("dtype", "auto")),
                device_map=str(item.get("device_map", "auto")),
                load_in_8bit=bool(item.get("load_in_8bit", False)),
                load_in_4bit=bool(item.get("load_in_4bit", False)),
                use_flash_attention_2=bool(item.get("use_flash_attention_2", False)),
                enable_thinking=enable_thinking,
            )
        )

    # Parse benchmarks
    benchmarks = [str(name).lower() for name in raw.get("benchmarks", [])]
    unknown = sorted(set(benchmarks) - SUPPORTED_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not benchmarks:
        raise ValueError(f"At least one benchmark is required. Options: {sorted(SUPPORTED_BENCHMARKS)}")

    # Parse generation parameters
    gen = raw.get("generation", {}) or {}
    if not isinstance(gen, dict):
        raise ValueError("generation must be a mapping.")

    # Apply Qwen best practices based on thinking mode
    enable_thinking_global = gen.get("enable_thinking")
    if enable_thinking_global is None:
        # If not specified, use model-specific defaults
        pass

    return EvalConfig(
        models=models,
        benchmarks=benchmarks,
        output_dir=str(raw.get("output_dir", "./eval_outputs")),
        cache_dir=raw.get("cache_dir", "./cache"),
        split=str(raw.get("split", "test")),
        max_samples=raw.get("max_samples"),
        seed=int(raw.get("seed", 42)),
        # Generation parameters with Qwen defaults
        max_new_tokens=int(gen.get("max_new_tokens", 32768)),
        temperature=float(gen.get("temperature", 1.0)),
        top_p=float(gen.get("top_p", 0.95)),
        top_k=int(gen.get("top_k", 20)),
        presence_penalty=float(gen.get("presence_penalty", 1.5)),
        repetition_penalty=float(gen.get("repetition_penalty", 1.0)),
        batch_size=int(gen.get("batch_size", 1)),
        benchmark_options=raw.get("benchmark_options", {}) or {},
    )