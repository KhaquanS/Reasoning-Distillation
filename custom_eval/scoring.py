"""
Scoring utilities for evaluating model outputs.
"""

import math
import re
from fractions import Fraction
from typing import Optional

from custom_eval.parsing import JSON_ANSWER_RE, find_choice_letter, last_boxed


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = str(text).strip()
    
    # Remove boxed markers
    boxed = last_boxed(text)
    if boxed:
        text = boxed
    
    # Remove JSON answer markers
    json_match = JSON_ANSWER_RE.search(text)
    if json_match:
        text = json_match.group(1)
    
    # Basic normalization
    text = text.lower().strip()
    text = re.sub(r"^(final\s*answer\s*:)", "", text).strip()
    text = re.sub(r"^(final\s*response\s*:)", "", text).strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .\"'")
    
    return text


def normalize_choice(text: str) -> str:
    """Normalize multiple-choice answer. Only a clearly marked label counts, never a stray letter in prose."""
    text = str(text).strip()
    text = last_boxed(text) or text
    json_match = JSON_ANSWER_RE.search(text)
    if json_match:
        text = json_match.group(1).strip()

    letter = find_choice_letter(text)
    if letter:
        return letter

    # A lone lowercase label, e.g. "b"
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    return normalize_text(text).upper()


def get_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text."""
    text = normalize_text(text)
    text = text.replace("$", "").replace("%", "")
    
    # Try fraction parsing
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    
    # Try regex for numbers
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    
    try:
        return float(match.group(0))
    except ValueError:
        return None


def exact_match(prediction: str, answer: str) -> bool:
    """Exact string match after normalization."""
    return normalize_text(prediction) == normalize_text(answer)


def numeric_or_exact_match(prediction: str, answer: str) -> bool:
    """Numeric match with tolerance, or exact string match."""
    pred_num = get_numeric_value(prediction)
    ans_num = get_numeric_value(answer)
    
    if pred_num is not None and ans_num is not None:
        return math.isclose(pred_num, ans_num, rel_tol=1e-6, abs_tol=1e-6)
    
    return exact_match(prediction, answer)


def choice_match(prediction: str, answer: str) -> bool:
    """Multiple-choice match."""
    return normalize_choice(prediction) == normalize_choice(answer)