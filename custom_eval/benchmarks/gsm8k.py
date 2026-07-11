"""GSM8K benchmark."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import numeric_or_exact_match


def _scrape_answer(answer):
    """Extract answer from GSM8K format."""
    return str(answer).split("####")[-1].strip()


def load(cache_dir=None, split="test", max_samples=None, **_):
    """Load GSM8K dataset."""
    ds, source = try_load_dataset(
        [{"path": "openai/gsm8k", "name": "main", "splits": [split, "test", "train"]}],
        cache_dir=cache_dir,
        split=split,
    )
    
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            examples.append(
                EvalExample(
                    str(i),
                    str(row["question"]),
                    _scrape_answer(row["answer"]),
                    {"source": source},
                )
            )
    else:
        # Fallback example
        examples = [
            EvalExample(
                "gsm8k_fallback_0",
                "Janet has 16 apples. She gives 5 to Tom and buys 8 more. How many apples does she have?",
                "19",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    
    return Benchmark("gsm8k", limit_examples(examples, max_samples), numeric_or_exact_match)