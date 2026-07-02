import random

from custom_eval.benchmarks.base import Benchmark, EvalExample
from custom_eval.benchmarks.loaders import limit_examples, try_load_dataset
from custom_eval.scoring import choice_match


LABELS = ["A", "B", "C", "D"]


def load(cache_dir=None, split="train", max_samples=None, seed=42, subset="gpqa_diamond", **_):
    ds, source = try_load_dataset(
        [{"path": "Idavidrein/gpqa", "name": subset, "splits": [split, "train"]}],
        cache_dir=cache_dir,
        split=split,
    )
    rng = random.Random(seed)
    examples = []
    if ds is not None:
        for i, row in enumerate(ds):
            correct = row.get("Correct Answer") or row.get("correct_answer")
            incorrect = [
                row.get("Incorrect Answer 1") or row.get("incorrect_answer_1"),
                row.get("Incorrect Answer 2") or row.get("incorrect_answer_2"),
                row.get("Incorrect Answer 3") or row.get("incorrect_answer_3"),
            ]
            choices = [correct] + incorrect
            rng.shuffle(choices)
            answer = LABELS[choices.index(correct)]
            choice_text = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
            question = row.get("Question") or row.get("question")
            examples.append(EvalExample(str(i), f"{question}\n\nChoices:\n{choice_text}", answer, {"source": source}))
    else:
        examples = [
            EvalExample(
                "gpqa_fallback_0",
                "Which particle has a negative electric charge?\n\nChoices:\nA. proton\nB. neutron\nC. electron\nD. photon",
                "C",
                {"source": "embedded_fallback", "load_errors": source.get("errors", [])},
            )
        ]
    return Benchmark("gpqa", limit_examples(examples, max_samples), choice_match)

