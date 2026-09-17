"""
Answer extraction from raw model outputs.

extract_answer() first handles thinking:
    unfinished_thinking  the prompt opened a <think> block but the output never closed it
                         (reasoning was cut off by the token limit); the answer is ""
    Otherwise everything up to the last </think> is dropped. If nothing follows it,
    the text before it is parsed instead.

It then tries these rules in order and reports which one matched:
    answer_tag     closed <answer>...</answer> blocks, latest first; then a trailing unclosed <answer>
    boxed          content of the last \\boxed{...}, nested braces allowed
    json           the last "answer": "..." pair
    phrase         the last "final answer: X" / "the answer is X" / "answer: X"
A rule only counts if it yields a usable answer (a choice label for multiple choice,
a number for gsm8k / aime25); otherwise the next rule is tried. Then the fallbacks:
    last_number    last number in the output (gsm8k / aime25)
    bare_letter    the last line is just a choice label, e.g. "B" or "**(B)**" (multiple choice)
    choice_phrase  the last two lines name a label, e.g. "The correct choice is **C**." (multiple choice)
    last_line      last non-empty line (math500 and other benchmarks)
    no_answer      nothing found; the answer is ""
"""

import re
from typing import List, Optional, Tuple

MATH_BENCHMARKS = {"math500", "aime25", "gsm8k"}
NUMERIC_BENCHMARKS = {"aime25", "gsm8k"}
CHOICE_BENCHMARKS = {"arc-c", "mmlu", "gpqa", "hellaswag"}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
# "answer": "B" / 'answer': 'B' / "answer": B
JSON_ANSWER_RE = re.compile(r"""["']answer["']\s*:\s*["']?([^"'\n}]+)""", re.IGNORECASE)
# The answer may sit on the next line, after markdown: "**Final Answer:**\nD"
PHRASE_RE = re.compile(
    r"(?:final\s+answer[\s*_]*(?:is|:)|the\s+(?:correct\s+)?answer\s+is|answer[\s*_]*:)[\s*_:]*([^\s*_][^\n]*)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CODE_FENCE_RE = re.compile(r"^\s*```\w*\s*$")
SHORT_SPAN = 60  # a one-line span this short may be read as a plain number

# Choice labels are uppercase A-E, or 1-5 (some ARC questions use digits).
# MARKUP is what may sit around a label: spaces, markdown, quotes, brackets.
_MARKUP = r"""[\s*_`"'(\[]*"""
_LABEL = r"([A-E1-5])(?![A-Za-z0-9])"
_IS_OR_COLON = r"(?:[\s*_]*(?i:is)|[\s*_]*:)?"
# "the answer is B", "**Answer:** (C)", "answer\nA"
ANSWER_LETTER_RE = re.compile(r"(?i:answer)" + _IS_OR_COLON + _MARKUP + _LABEL)
# "the correct choice is **C**", "the most plausible ending is D", "the answer is option B",
# "Choose C". A bare "Choice D is wrong" does not count.
OPTION_LETTER_RE = re.compile(
    r"(?i:(?:correct|right|best|final|most\s+\w+|answer\s+is)\s+(?:option|choice|ending|answer)"
    r"|(?:choose|select|pick)(?:\s+(?:option|choice))?)"
    + _IS_OR_COLON + _MARKUP + _LABEL
)
# "**C** is the most plausible", "D is the correct answer"
LETTER_IS_RE = re.compile(
    r"(?<![A-Za-z0-9])[*_`\"'(\[]*([A-E])[)\]*_`\"']*\s+is\s+(?:the\s+)?"
    r"(?i:correct|right|best|most\s+(?:plausible|likely|appropriate|logical))"
)
# A label on its own: "B", "(B)", "B.", "B)", "B: 4", "**B**", "\"B\"", "Option B."
LEADING_LETTER_RE = re.compile(
    r"""^[\s*_`"'(\[]*(?:(?i:option|choice)\s*)?([A-E1-5])[)\]*_`"']*(?:[.:),]|\s*$)"""
)


def strip_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split output at the last </think>. Returns (thinking, rest); thinking is None if there is no </think>."""
    idx = text.rfind(THINK_CLOSE)
    if idx == -1:
        return None, text
    thinking = text[:idx].strip()
    if thinking.startswith(THINK_OPEN):
        thinking = thinking[len(THINK_OPEN):].strip()
    return thinking, text[idx + len(THINK_CLOSE):]


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, handling nested braces. None if absent or unclosed."""
    start = text.rfind("\\boxed{")
    while start != -1:
        i = start + len("\\boxed{")
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start + len("\\boxed{"):i - 1].strip()
        # Unclosed (e.g. truncated) box: try an earlier one
        start = text.rfind("\\boxed{", 0, start)
    return None


def _last(pattern: re.Pattern, text: str) -> Optional[str]:
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def find_choice_letter(text: str) -> Optional[str]:
    """Strictly find a choice label in a span. Never picks a stray letter out of prose."""
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    for pattern in (ANSWER_LETTER_RE, OPTION_LETTER_RE, LETTER_IS_RE):
        letter = _last(pattern, text)
        if letter:
            return letter
    match = LEADING_LETTER_RE.match(text)
    return match.group(1) if match else None


def _last_number(text: str) -> Optional[str]:
    number = _last(NUMBER_RE, text)
    return number.replace(",", "") if number else None


def _refine(span: str, benchmark_name: str) -> Optional[str]:
    """Narrow an extracted span down to the answer itself. None if the span holds no usable answer."""
    span = span.strip()
    if not span:
        return None
    if benchmark_name in CHOICE_BENCHMARKS:
        for inner in (last_boxed(span), _last(JSON_ANSWER_RE, span), span):
            letter = find_choice_letter(inner) if inner else None
            if letter:
                return letter
        return None
    if benchmark_name in MATH_BENCHMARKS:
        inner = last_boxed(span) or _last(PHRASE_RE, span)
        if benchmark_name in NUMERIC_BENCHMARKS:
            if inner is None and ("\n" in span or len(span) > SHORT_SPAN):
                return None  # a paragraph of working, not an answer
            span = (inner or span).strip()
            # "9 * 2 = 18" or "$18 per day" -> "18"
            return span if NUMBER_RE.fullmatch(span) else _last_number(span)
        span = (inner or span).strip()
    return span or None


def _answer_tags(text: str) -> List[str]:
    """Closed <answer> blocks latest first, then a trailing unclosed <answer> (often cut off)."""
    closed = list(ANSWER_TAG_RE.finditer(text))
    tags = [m.group(1) for m in reversed(closed)]
    opens = list(ANSWER_OPEN_RE.finditer(text))
    if opens and (not closed or opens[-1].start() >= closed[-1].end()):
        tags.append(text[opens[-1].end():])
    return tags


def _choice_fallback(text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not CODE_FENCE_RE.match(line)]
    if not lines:
        return "", "no_answer"
    match = LEADING_LETTER_RE.match(lines[-1])
    if match:
        return match.group(1), "bare_letter"
    letter = find_choice_letter("\n".join(lines[-2:]))
    if letter:
        return letter, "choice_phrase"
    return "", "no_answer"


def extract_answer(text: str, benchmark_name: str, thinking_opened: bool = False) -> Tuple[str, str]:
    """
    Return (answer, rule) for a raw model output. See module docstring for the rules.

    thinking_opened: the prompt ended with an open <think> block, so the output must
    close it with </think> before giving an answer.
    """
    text = text or ""
    benchmark_name = benchmark_name.lower()

    if thinking_opened and THINK_CLOSE not in text:
        return "", "unfinished_thinking"
    thinking, rest = strip_thinking(text)
    if thinking is not None and not rest.strip():
        rest = thinking

    candidates = [("answer_tag", tag) for tag in _answer_tags(rest)] + [
        ("boxed", last_boxed(rest)),
        ("json", _last(JSON_ANSWER_RE, rest)),
        ("phrase", _last(PHRASE_RE, rest)),
    ]
    for rule, value in candidates:
        answer = _refine(value, benchmark_name) if value else None
        if answer:
            return answer, rule

    if benchmark_name in NUMERIC_BENCHMARKS:
        number = _last_number(rest)
        return (number, "last_number") if number else ("", "no_answer")
    if benchmark_name in CHOICE_BENCHMARKS:
        return _choice_fallback(rest)
    lines = [line.strip() for line in rest.splitlines() if line.strip()]
    return (lines[-1], "last_line") if lines else ("", "no_answer")
