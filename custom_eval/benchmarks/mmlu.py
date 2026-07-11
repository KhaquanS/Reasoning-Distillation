"""MMLU benchmark."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import choice_match


LABELS = ["A", "B", "C", "D"]


def load(cache_dir=None, split="test", max_samples=None, subject="all", **_):
    """Load MMLU dataset."""
    ds, source = try_load_dataset(
        [
            {"path": "cais/mmlu", "name": subject, "splits": [split, "test", "validation", "dev"]},
            {"path": "lukaemon/mmlu", "name": subject, "splits": [split, "test", "validation"]},
        ],
        cache_dir=cache_dir,
        split=split,
    )
    
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            choices = row.get("choices") or [row.get(f"choice_{j}") for j in range(4)]
            choice_text = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
            answer = row.get("answer")
            if isinstance(answer, int):
                answer = LABELS[answer]
            examples.append(
                EvalExample(
                    str(i),
                    f"{row['question']}\n\nChoices:\n{choice_text}",
                    str(answer),
                    {"source": source},
                )
            )
    else:
        # Fallback example
        examples = [
            EvalExample(
                "mmlu_fallback_0",
                "What is the capital of France?\n\nChoices:\nA. Berlin\nB. Madrid\nC. Paris\nD. Rome",
                "C",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    
    return Benchmark("mmlu", limit_examples(examples, max_samples), choice_match)