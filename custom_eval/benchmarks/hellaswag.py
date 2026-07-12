"""HellaSwag benchmark."""

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import choice_match


LABELS = ["A", "B", "C", "D"]


def load(cache_dir=None, split="validation", max_samples=None, **_):
    """Load HellaSwag dataset."""
    ds, source = try_load_dataset(
        [{"path": "Rowan/hellaswag", "splits": [split, "validation", "train", "test"]}],
        cache_dir=cache_dir,
        split=split,
    )
    
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            # Get the label; it could be int or string
            label_val = row.get("label")
            if label_val is None or label_val == "":
                # Skip examples with missing label
                continue
            try:
                label_idx = int(label_val)
            except (ValueError, TypeError):
                # If conversion fails, skip
                continue
            
            if not (0 <= label_idx < len(LABELS)):
                continue
            
            endings = row["endings"]
            choices = "\n".join(f"{label}. {ending}" for label, ending in zip(LABELS, endings))
            context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip() or row.get("ctx", "")
            question = f"Choose the most plausible ending.\n\nContext: {context}\n\nChoices:\n{choices}"
            answer = LABELS[label_idx]
            examples.append(
                EvalExample(
                    str(row.get("ind", i)),
                    question,
                    answer,
                    {"source": source},
                )
            )
    else:
        # Fallback example
        examples = [
            EvalExample(
                "hellaswag_fallback_0",
                "Choose the most plausible ending.\n\nContext: A person opens an umbrella because\n\nChoices:\nA. it is raining outside.\nB. the sun turned into music.\nC. they are reading a book.\nD. the floor is asleep.",
                "A",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    
    return Benchmark("hellaswag", limit_examples(examples, max_samples), choice_match)