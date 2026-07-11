"""Base classes for benchmarks."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class EvalExample:
    """A single evaluation example."""
    id: str
    question: str
    answer: str
    metadata: Optional[Dict] = None


@dataclass
class Benchmark:
    """A benchmark dataset."""
    name: str
    examples: list[EvalExample]
    scorer: Callable[[str, str], bool]