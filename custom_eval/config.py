from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    tokenizer: str | None = None
    trust_remote_code: bool = True
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_8bit: bool = False


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    benchmarks: list[str]
    output_dir: str = "./eval_outputs"
    cache_dir: str | None = "./cache"
    split: str = "test"
    max_samples: int | None = None
    seed: int = 42
    cot: bool = False
    max_toks: int | None = None
    pass_at_k: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    batch_size: int = 1
    benchmark_options: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path) -> EvalConfig:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

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
        models.append(
            ModelSpec(
                name=item.get("name") or Path(str(checkpoint)).name,
                checkpoint=str(checkpoint),
                tokenizer=item.get("tokenizer"),
                trust_remote_code=bool(item.get("trust_remote_code", True)),
                dtype=str(item.get("dtype", "auto")),
                device_map=str(item.get("device_map", "auto")),
                load_in_8bit=bool(item.get("load_in_8bit", False)),
            )
        )

    benchmarks = [str(name).lower() for name in raw.get("benchmarks", [])]
    unknown = sorted(set(benchmarks) - SUPPORTED_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not benchmarks:
        raise ValueError(f"At least one benchmark is required. Options: {sorted(SUPPORTED_BENCHMARKS)}")

    generation = raw.get("generation", {}) or {}
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping.")

    cot = bool(generation.get("cot", raw.get("cot", False)))
    max_toks = generation.get("max_toks", raw.get("max_toks"))
    if cot and max_toks is None:
        raise ValueError("generation.max_toks is required when generation.cot is true.")
    if max_toks is not None and int(max_toks) <= 0:
        raise ValueError("generation.max_toks must be positive.")

    pass_at_k = int(generation.get("pass_at_k", raw.get("pass_at_k", 1)))
    if pass_at_k <= 0:
        raise ValueError("generation.pass_at_k must be positive.")

    benchmark_options = raw.get("benchmark_options", {}) or {}
    if not isinstance(benchmark_options, dict):
        raise ValueError("benchmark_options must be a mapping.")

    return EvalConfig(
        models=models,
        benchmarks=benchmarks,
        output_dir=str(raw.get("output_dir", "./eval_outputs")),
        cache_dir=raw.get("cache_dir", "./cache"),
        split=str(raw.get("split", "test")),
        max_samples=raw.get("max_samples"),
        seed=int(raw.get("seed", 42)),
        cot=cot,
        max_toks=int(max_toks) if max_toks is not None else None,
        pass_at_k=pass_at_k,
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_new_tokens=int(generation.get("max_new_tokens", 128)),
        batch_size=int(generation.get("batch_size", 1)),
        benchmark_options=benchmark_options,
    )

