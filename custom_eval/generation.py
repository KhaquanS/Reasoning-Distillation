"""
Generation utilities for Qwen models with proper chat formatting, supporting batching and pass@k.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from custom_eval.prompts.qwen_formatter import QwenChatFormatter
from custom_eval.prompts.templates import build_prompt


@dataclass
class GeneratedCandidate:
    """A single generated candidate with metadata."""
    prompt: str
    raw_output: str
    final_response: str
    thinking_content: Optional[str] = None


# ----------------------------------------------------------------------------
# Answer extraction helpers
# ----------------------------------------------------------------------------

def extract_final_answer_from_boxed(text: str) -> Optional[str]:
    """Extract answer from \boxed{} format."""
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_json_answer(text: str) -> Optional[str]:
    """Extract answer from JSON format like {"answer": "C"}."""
    import json
    try:
        # Look for inline JSON with "answer" key
        match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try parsing the whole text as JSON
        data = json.loads(text)
        if "answer" in data:
            return str(data["answer"]).strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def extract_final_response(
    text: str,
    benchmark_name: str = "default",
    enable_thinking: bool = False,
) -> str:
    """
    Extract the final answer from model output based on benchmark type.
    """
    if not text:
        return ""

    # Remove thinking content if present
    if enable_thinking:
        text = QwenChatFormatter.extract_response(text)

    text = text.strip()

    # Math benchmarks: look for \boxed{...}
    if benchmark_name in {"math500", "aime25", "gsm8k"}:
        ans = extract_final_answer_from_boxed(text)
        if ans:
            return ans

    # Multiple-choice benchmarks: look for JSON answer
    if benchmark_name in {"arc-c", "mmlu", "gpqa", "hellaswag"}:
        ans = extract_json_answer(text)
        if ans:
            return ans

    # Generic fallback patterns
    patterns = [
        r"final\s+answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
        r"therefore\s*,\s*(.+)$",
        r"so\s*,\s*(.+)$",
        r"the answer is\s*(.+)",
        r"^([A-D])$",                     # single letter
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            ans = match.group(1).strip()
            if ans:
                return ans

    # Last resort: return the last non‑empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


# ----------------------------------------------------------------------------
# Core generation functions
# ----------------------------------------------------------------------------

def _format_prompt(
    question: str,
    benchmark_name: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    system_prompt: Optional[str] = None,
) -> str:
    """Build and apply the chat template to a single question."""
    prompt = build_prompt(question, benchmark_name)
    messages = QwenChatFormatter.format_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_candidates_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    benchmark_name: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    pass_at_k: int = 1,
    system_prompt: Optional[str] = None,
    max_input_length: int = 4096,
) -> List[List[GeneratedCandidate]]:
    """
    Generate candidates for a batch of questions, with support for pass@k.

    Returns:
        A list of lists: for each question, a list of GeneratedCandidate
        of length `pass_at_k`.
    """
    if not questions:
        return []

    # 1. Format each question into a prompt
    prompts = [
        _format_prompt(q, benchmark_name, tokenizer, enable_thinking, system_prompt)
        for q in questions
    ]

    # 2. Tokenize with padding
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    # Store original sequence lengths (number of non-pad tokens per prompt)
    orig_lengths = attention_mask.sum(dim=1).tolist()  # list of ints

    # 3. Repeat each input `pass_at_k` times for multiple candidates
    if pass_at_k > 1:
        input_ids = input_ids.repeat_interleave(pass_at_k, dim=0)
        attention_mask = attention_mask.repeat_interleave(pass_at_k, dim=0)
        orig_lengths = [l for l in orig_lengths for _ in range(pass_at_k)]

    # 4. Generate
    # NOTE: presence_penalty is not supported by transformers' generate().
    # Only repetition_penalty is available.
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    # 5. Decode each output, cutting off the input part
    raw_outputs = []
    for i, (inp, out) in enumerate(zip(input_ids, output_ids)):
        input_len = orig_lengths[i]
        gen_tokens = out[input_len:]               # generated part only
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        raw_outputs.append(raw)

    # 6. Build Candidate objects
    candidates_flat = []
    for i, raw in enumerate(raw_outputs):
        # Which original question does this belong to?
        question_idx = i // pass_at_k
        prompt_used = prompts[question_idx]

        final_response = extract_final_response(
            raw,
            benchmark_name=benchmark_name,
            enable_thinking=enable_thinking,
        )
        thinking_content = None
        if enable_thinking:
            think_pattern = re.compile(r"<think\s*>\s*(.*?)\s*</think\s*>", re.IGNORECASE | re.DOTALL)
            match = think_pattern.search(raw)
            if match:
                thinking_content = match.group(1).strip()

        candidates_flat.append(
            GeneratedCandidate(
                prompt=prompt_used,
                raw_output=raw,
                final_response=final_response,
                thinking_content=thinking_content,
            )
        )

    # 7. Group by original question
    grouped = []
    for q_idx in range(len(questions)):
        start = q_idx * pass_at_k
        end = start + pass_at_k
        grouped.append(candidates_flat[start:end])

    return grouped


def generate_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    benchmark_name: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    pass_at_k: int = 1,
    system_prompt: Optional[str] = None,
    max_input_length: int = 4096,
) -> List[GeneratedCandidate]:
    """
    Single‑question wrapper around batched generation.
    """
    results = generate_candidates_batch(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        benchmark_name=benchmark_name,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pass_at_k=pass_at_k,
        system_prompt=system_prompt,
        max_input_length=max_input_length,
    )
    return results[0] if results else []