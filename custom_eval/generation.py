"""
Generation utilities for Qwen models with proper chat formatting, supporting batching and pass@k.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

from custom_eval.prompts.qwen_formatter import QwenChatFormatter
from custom_eval.math_scoring import extract_boxed_answer

@dataclass
class GeneratedCandidate:
    """A single generated candidate with metadata."""
    prompt: str
    raw_output: str
    final_response: str
    thinking_content: Optional[str] = None
    generated_tokens: Optional[int] = None
    hit_token_limit: Optional[bool] = None


# ----------------------------------------------------------------------------
# Answer extraction helpers
# ----------------------------------------------------------------------------

def extract_final_answer_from_boxed(text: str) -> Optional[str]:
    """Extract the last balanced boxed answer, including nested LaTeX."""
    return extract_boxed_answer(text)


def extract_json_answer(text: str) -> Optional[str]:
    """Extract answer from JSON format like {"answer": "C"}."""
    import json
    try:
        match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
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
    model_type: str = "qwen",
) -> str:
    """
    Extract the final answer from model output based on benchmark type and model_type.
    """
    if not text:
        return ""

    if benchmark_name in {"math500", "aime25"}:
        # The opening think tag may be in the prompt. Do not score reasoning-only
        # completions, including those truncated before a final answer.
        if model_type == "qwen" and enable_thinking:
            parts = re.split(r"</think\s*>", text, flags=re.IGNORECASE)
            if len(parts) == 1:
                return ""
            text = parts[-1]
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text,
                      flags=re.IGNORECASE | re.DOTALL)
        text = re.split(r"<think\b[^>]*>", text, flags=re.IGNORECASE)[0]
        return extract_boxed_answer(text) or ""

    # Remove thinking block only for Qwen models with thinking enabled
    if model_type == "qwen" and enable_thinking:
        text = QwenChatFormatter.extract_response(text)

    text = text.strip()

    if benchmark_name in {"math500", "aime25", "gsm8k"}:
        ans = extract_final_answer_from_boxed(text)
        if ans:
            return ans

    if benchmark_name in {"arc-c", "mmlu", "gpqa", "hellaswag"}:
        ans = extract_json_answer(text)
        if ans:
            return ans

    patterns = [
        r"final\s+answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
        r"therefore\s*,\s*(.+)$",
        r"so\s*,\s*(.+)$",
        r"the answer is\s*(.+)",
        r"^([A-D])$",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            ans = match.group(1).strip()
            if ans:
                return ans

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


# ----------------------------------------------------------------------------
# Token progress callback
# ----------------------------------------------------------------------------

class TokenProgressCallback(StoppingCriteria):
    """A stopping criteria that updates a tqdm progress bar per generation step."""
    def __init__(self, pbar: tqdm, total: int):
        self.pbar = pbar
        self.total = total
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.step += 1
        # Update progress bar, but cap at total to avoid overshoot
        if self.step <= self.total:
            self.pbar.update(1)
        return False  # never stop early (we rely on other criteria)

    def close(self):
        self.pbar.close()


# ----------------------------------------------------------------------------
# Core generation functions
# ----------------------------------------------------------------------------

def build_messages(
    prompt: str,
    system_prompt: Optional[str],
    model_type: str,
    enable_thinking: bool,
) -> List[dict]:
    """
    Build a message list suitable for the tokenizer's chat template.
    For Qwen, use the QwenChatFormatter; for others, build a simple system/user pair.
    """
    if model_type == "qwen":
        return QwenChatFormatter.format_messages(prompt, system_prompt, enable_thinking)
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


def _format_prompt(
    question: str,
    benchmark_name: str,
    tokenizer: PreTrainedTokenizer,
    model_type: str,
    enable_thinking: bool,
    system_prompt: Optional[str] = None,
) -> str:
    """Build and apply the chat template to a single question."""
    from custom_eval.prompts.templates import build_prompt

    prompt = build_prompt(question, benchmark_name)
    messages = build_messages(prompt, system_prompt, model_type, enable_thinking)
    template_kwargs = {"enable_thinking": enable_thinking} if model_type == "qwen" else {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def generate_candidates_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    benchmark_name: str,
    model_type: str = "qwen",
    enable_thinking: bool = True,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    pass_at_k: int = 1,
    system_prompt: Optional[str] = None,
    max_input_length: int = 4096,
    show_progress: bool = False,
) -> List[List[GeneratedCandidate]]:
    """
    Generate candidates for a batch of questions, with support for pass@k.

    If show_progress is True, a tqdm progress bar will be shown for token generation
    within each batch.
    """
    if not questions:
        return []

    # Format each question into a prompt
    prompts = [
        _format_prompt(q, benchmark_name, tokenizer, model_type, enable_thinking, system_prompt)
        for q in questions
    ]

    # Tokenize with padding
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    # Generated sequences include the entire padded input, not just non-pad tokens.
    input_width = input_ids.shape[1]

    if pass_at_k > 1:
        input_ids = input_ids.repeat_interleave(pass_at_k, dim=0)
        attention_mask = attention_mask.repeat_interleave(pass_at_k, dim=0)

    # Generation kwargs
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

    # Set up progress bar if requested
    progress_callback = None
    if show_progress:
        # Create a pbar with total = max_new_tokens (or at least 1)
        pbar = tqdm(
            total=max_new_tokens,
            desc=f"Generating {len(questions)} questions",
            unit="tok",
            leave=False,
        )
        progress_callback = TokenProgressCallback(pbar, max_new_tokens)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([progress_callback])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    if progress_callback is not None:
        progress_callback.close()  # close the pbar

    # Decode and group candidates
    raw_outputs = []
    output_lengths = []
    limit_flags = []
    for out in output_ids:
        gen_tokens = out[input_width:]
        token_list = gen_tokens.tolist() if hasattr(gen_tokens, "tolist") else list(gen_tokens)
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        # Ignore EOS and batch padding when reporting answer length.
        end = next((i for i, token in enumerate(token_list) if int(token) in eos_ids), len(gen_tokens))
        output_lengths.append(end)
        limit_flags.append(end == len(gen_tokens) and end >= max_new_tokens)
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        raw_outputs.append(raw)

    candidates_flat = []
    for i, raw in enumerate(raw_outputs):
        question_idx = i // pass_at_k
        prompt_used = prompts[question_idx]

        final_response = extract_final_response(
            raw,
            benchmark_name=benchmark_name,
            enable_thinking=enable_thinking,
            model_type=model_type,
        )
        thinking_content = None
        if model_type == "qwen" and enable_thinking:
            # The opening tag may already be part of the input chat template.
            parts = re.split(r"</think\s*>", raw, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                thinking_content = re.sub(
                    r"^\s*<think\s*>\s*", "", parts[0], flags=re.IGNORECASE
                ).strip()

        candidates_flat.append(
            GeneratedCandidate(
                prompt=prompt_used,
                raw_output=raw,
                final_response=final_response,
                thinking_content=thinking_content,
                generated_tokens=output_lengths[i],
                hit_token_limit=limit_flags[i],
            )
        )

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
    model_type: str = "qwen",
    enable_thinking: bool = True,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    pass_at_k: int = 1,
    system_prompt: Optional[str] = None,
    max_input_length: int = 4096,
    show_progress: bool = False,
) -> List[GeneratedCandidate]:
    """
    Single‑question wrapper around batched generation.
    """
    results = generate_candidates_batch(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        benchmark_name=benchmark_name,
        model_type=model_type,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pass_at_k=pass_at_k,
        system_prompt=system_prompt,
        max_input_length=max_input_length,
        show_progress=show_progress,
    )
    return results[0] if results else []