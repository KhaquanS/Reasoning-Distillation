"""Final-answer scoring for MATH-500 and AIME; never mine numbers from reasoning."""

import re
from functools import lru_cache


def extract_boxed_answer(text):
    """Return the last boxed/fboxed answer, balancing nested and escaped braces.

    An incomplete final box fails extraction rather than reusing an earlier box.
    """
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip() or None
        i += 1
    return None


def require_math_verify():
    try:
        import math_verify
    except ImportError as exc:
        raise ImportError(
            "MATH-500 scoring requires math-verify==0.8.0. "
            "Install requirements-eval.txt before evaluation."
        ) from exc
    return math_verify


@lru_cache(maxsize=1024)
def _parse_answer(answer):
    math_verify = require_math_verify()
    return math_verify.parse(
        "\\boxed{" + answer + "}",
        extraction_config=[math_verify.LatexExtractionConfig(boxed_match_priority=0)],
        extraction_mode="first_match",
        fallback_mode="no_fallback",
        parsing_timeout=5,
    )


def math_equivalent(prediction, answer):
    """Compare extracted answers with Math-Verify: gold first, strict symbols."""
    if not prediction or not str(prediction).strip():
        return False
    math_verify = require_math_verify()
    gold, pred = str(answer).strip(), str(prediction).strip()
    # Preserve literal text answers such as \\text{Evelyn}.
    if pred == gold:
        return True
    parsed_gold, parsed_pred = _parse_answer(gold), _parse_answer(pred)
    return bool(parsed_gold and parsed_pred and math_verify.verify(
        parsed_gold, parsed_pred, strict=True, timeout_seconds=5,
    ))


def aime_integer(answer):
    """AIME answers are single integers from 0 to 999; accept leading zeros."""
    text = str(answer).strip()
    if not re.fullmatch(r"[0-9]{1,3}", text):
        return None
    return int(text)


def aime_exact_match(prediction, answer):
    pred, gold = aime_integer(prediction), aime_integer(answer)
    return pred is not None and gold is not None and pred == gold
