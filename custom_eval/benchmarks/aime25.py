from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import numeric_or_exact_match


def load(cache_dir=None, split="test", max_samples=None, **_):
    ds, source = try_load_dataset(
        [
            {"path": "yentinglin/aime_2025", "splits": [split, "test", "train"]},
            {"path": "Maxwell-Jia/AIME_2025", "splits": [split, "test", "train"]},
        ],
        cache_dir=cache_dir,
        split=split,
    )
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            question = row.get("problem") or row.get("question") or row.get("prompt")
            answer = row.get("answer") or row.get("final_answer")
            examples.append(EvalExample(str(i), str(question), str(answer), {"source": source}))
    else:
        examples = [
            EvalExample(
                "aime25_fallback_0",
                "Find the least positive integer n such that n leaves a remainder of 1 when divided by 2 and 3.",
                "1",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("aime25", limit_examples(examples, max_samples), numeric_or_exact_match)

