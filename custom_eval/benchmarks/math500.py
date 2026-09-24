"""The fixed 500-question test subset published by HuggingFaceH4."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import load_required_dataset, limit_examples
from custom_eval.math_scoring import math_equivalent, require_math_verify


def load(cache_dir=None, split="test", max_samples=None, **_):
    require_math_verify()
    ds, source = load_required_dataset("HuggingFaceH4/MATH-500", 500, cache_dir, split)
    examples = []
    for row in ds:
        if not row.get("problem") or row.get("answer") is None or not row.get("unique_id"):
            raise ValueError("MATH-500 requires problem, answer, and unique_id fields.")
        examples.append(EvalExample(
            str(row["unique_id"]), row["problem"], str(row["answer"]),
            {"source": source, "subject": row.get("subject"), "level": row.get("level")},
        ))
    if len({ex.id for ex in examples}) != len(examples):
        raise ValueError("Duplicate MATH-500 problem IDs.")
    return Benchmark("math500", limit_examples(examples, max_samples), math_equivalent)
