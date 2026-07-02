from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import choice_match


def _format_choices(labels, texts):
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def load(cache_dir=None, split="test", max_samples=None, **_):
    ds, source = try_load_dataset(
        [{"path": "allenai/ai2_arc", "name": "ARC-Challenge", "splits": [split, "test", "validation"]}],
        cache_dir=cache_dir,
        split=split,
    )
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            question = f"{row['question']}\n\nChoices:\n{_format_choices(labels, texts)}"
            examples.append(EvalExample(str(row.get("id", i)), question, str(row["answerKey"]), {"source": source}))
    else:
        examples = [
            EvalExample(
                "arc_fallback_0",
                "Which property of a mineral can be determined just by looking at it?\n\nChoices:\nA. luster\nB. mass\nC. weight\nD. hardness",
                "A",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("arc-c", limit_examples(examples, max_samples), choice_match)

