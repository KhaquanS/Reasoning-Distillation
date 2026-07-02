import math
import re
from fractions import Fraction


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def normalize_text(text: str) -> str:
    text = str(text).strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.lower().strip()
    text = re.sub(r"^(final\s*response\s*:)", "", text).strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def normalize_choice(text: str) -> str:
    text = normalize_text(text)
    match = re.search(r"\b([a-e])\b", text)
    if match:
        return match.group(1).upper()
    if text and text[0] in "abcde":
        return text[0].upper()
    return text.upper()


def _numeric_value(text: str):
    text = normalize_text(text)
    text = text.replace("$", "").replace("%", "")
    try:
        return float(Fraction(text))
    except Exception:
        pass
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_text(prediction) == normalize_text(answer)


def numeric_or_exact_match(prediction: str, answer: str) -> bool:
    pred_num = _numeric_value(prediction)
    ans_num = _numeric_value(answer)
    if pred_num is not None and ans_num is not None:
        return math.isclose(pred_num, ans_num, rel_tol=1e-6, abs_tol=1e-6)
    return exact_match(prediction, answer)


def choice_match(prediction: str, answer: str) -> bool:
    return normalize_choice(prediction) == normalize_choice(answer)

