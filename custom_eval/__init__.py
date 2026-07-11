"""YAML-driven evaluation harness for distilled and Hugging Face LMs with Qwen support."""

from custom_eval.benchmarks import (
    aime25,
    arc_c,
    gpqa,
    gsm8k,
    hellaswag,
    math500,
    mmlu,
)
from custom_eval.prompts import qwen_formatter

__version__ = "2.0.0"