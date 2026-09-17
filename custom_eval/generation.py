"""
Generation utilities for Qwen models with proper chat formatting, supporting batching and pass@k.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

from custom_eval.parsing import THINK_OPEN, extract_answer, strip_thinking
from custom_eval.prompts.qwen_formatter import QwenChatFormatter

@dataclass
class GeneratedCandidate:
    """A single generated candidate with metadata."""
    prompt: str
    raw_output: str
    final_response: str
    thinking_content: Optional[str] = None
    parse_rule: str = "no_answer"  # which extraction rule produced final_response (see parsing.py)


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
    # enable_thinking is read by Qwen's chat template; other templates ignore it
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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

    # Decode and group candidates. Prompts are left-padded, so every row's
    # generated tokens start at the padded width, not at its unpadded length.
    prompt_width = input_ids.shape[1]
    raw_outputs = []
    for out in output_ids:
        gen_tokens = out[prompt_width:]
        raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        raw_outputs.append(raw)

    candidates_flat = []
    for i, raw in enumerate(raw_outputs):
        question_idx = i // pass_at_k
        prompt_used = prompts[question_idx]

        # The template opens <think> in the prompt, so the output only contains </think>
        thinking_opened = prompt_used.rstrip().endswith(THINK_OPEN)
        thinking_content, _ = strip_thinking(raw)
        final_response, parse_rule = extract_answer(raw, benchmark_name, thinking_opened)

        candidates_flat.append(
            GeneratedCandidate(
                prompt=prompt_used,
                raw_output=raw,
                final_response=final_response,
                thinking_content=thinking_content,
                parse_rule=parse_rule,
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