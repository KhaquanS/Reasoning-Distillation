"""
Scoring utilities for evaluating model outputs.
"""

import math
import re
from fractions import Fraction
from typing import Optional


# Regular expressions for answer extraction
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
JSON_ANSWER_RE = re.compile(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = str(text).strip()
    
    # Remove boxed markers
    boxed = BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    
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
    """Normalize multiple-choice answer."""
    text = normalize_text(text)
    
    # Look for single letter
    match = re.search(r"\b([a-e])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Check if it's a single character
    if text and len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    
    return text.upper()


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