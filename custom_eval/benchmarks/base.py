from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalExample:
    id: str
    question: str
    answer: str
    metadata: dict


@dataclass
class Benchmark:
    name: str
    examples: list[EvalExample]
    scorer: Callable[[str, str], bool]

