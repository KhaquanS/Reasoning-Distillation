"""MATH-500 benchmark."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import numeric_or_exact_match


def load(cache_dir=None, split="test", max_samples=None, **_):
    """Load MATH-500 dataset."""
    ds, source = try_load_dataset(
        [
            {"path": "HuggingFaceH4/MATH-500", "splits": [split, "test"]},
            {"path": "lighteval/MATH", "name": "all", "splits": [split, "test"]},
        ],
        cache_dir=cache_dir,
        split=split,
    )
    
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            question = row.get("problem") or row.get("question") or row.get("text")
            answer = row.get("answer") or row.get("solution") or row.get("final_answer")
            examples.append(
                EvalExample(
                    str(i),
                    str(question),
                    str(answer),
                    {"source": source},
                )
            )
    else:
        # Fallback example
        examples = [
            EvalExample(
                "math500_fallback_0",
                "Compute 12 * 13.",
                "156",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    
    return Benchmark("math500", limit_examples(examples, max_samples), numeric_or_exact_match)