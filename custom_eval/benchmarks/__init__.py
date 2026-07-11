"""Benchmark loaders and adapters."""

from custom_eval.benchmarks import aime25, arc_c, gpqa, gsm8k, hellaswag, math500, mmlu
from custom_eval.benchmarks.base import Benchmark, EvalExample


LOADERS = {
    "arc-c": arc_c.load,
    "math500": math500.load,
    "aime25": aime25.load,
    "gsm8k": gsm8k.load,
    "hellaswag": hellaswag.load,
    "mmlu": mmlu.load,
    "gpqa": gpqa.load,
}


def load_benchmark(name, cache_dir=None, split="test", max_samples=None, seed=42, options=None):
    """Load a benchmark by name."""
    options = options or {}
    return LOADERS[name](
        cache_dir=cache_dir,
        split=options.get("split", split),
        max_samples=options.get("max_samples", max_samples),
        seed=seed,
        **{k: v for k, v in options.items() if k not in {"split", "max_samples"}},
    )


__all__ = [
    "Benchmark",
    "EvalExample",
    "load_benchmark",
    "arc_c",
    "math500",
    "aime25",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "gpqa",
]