"""AIME 2025 I and II: the 30 problems in math-ai/aime25, test split."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import load_required_dataset, limit_examples
from custom_eval.math_scoring import aime_exact_match, aime_integer


def load(cache_dir=None, split="test", max_samples=None, **_):
    ds, source = load_required_dataset("math-ai/aime25", 30, cache_dir, split)
    examples = []
    for row in ds:
        if not row.get("problem") or row.get("answer") is None or row.get("id") is None:
            raise ValueError("AIME 2025 requires problem, answer, and id fields.")
        if aime_integer(row["answer"]) is None:
            raise ValueError(f"Invalid AIME gold answer for {row['id']}: {row['answer']!r}")
        examples.append(EvalExample(
            str(row["id"]), row["problem"], str(row["answer"]), {"source": source},
        ))
    if len({ex.id for ex in examples}) != len(examples):
        raise ValueError("Duplicate AIME 2025 problem IDs.")
    return Benchmark("aime25", limit_examples(examples, max_samples), aime_exact_match)
