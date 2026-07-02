import re
from dataclasses import dataclass

import torch


FINAL_RE = re.compile(r"final\s*response\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class GeneratedCandidate:
    prompt: str
    raw_output: str
    final_response: str


def extract_final_response(text: str) -> str:
    match = FINAL_RE.search(text.strip())
    if match:
        answer = match.group(1).strip()
        return answer.splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def build_prompt(question: str, cot: bool) -> str:
    if cot:
        return (
            f"{question}\n\n"
            "Think through the problem. Do not give the final answer yet."
        )
    return (
        f"{question}\n\n"
        "Return only the final answer on a new line exactly as:\n"
        "final response: {answer}"
    )


def _generate_once(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_candidates(
    model,
    tokenizer,
    question: str,
    cot: bool,
    max_toks: int | None,
    pass_at_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[GeneratedCandidate]:
    prompt = build_prompt(question, cot)
    candidates = []
    for _ in range(pass_at_k):
        if cot:
            reasoning = _generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_toks,
                temperature=temperature,
                top_p=top_p,
            )
            final_prompt = (
                f"{question}\n\nReasoning:\n{reasoning}\n\n"
                "Now return only the final answer on a new line exactly as:\n"
                "final response: {answer}"
            )
            final_text = _generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=final_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw_output = f"{reasoning}\n{final_text}".strip()
            candidate_prompt = f"{prompt}\n\n--- final prompt ---\n{final_prompt}"
        else:
            raw_output = _generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            candidate_prompt = prompt
        final_response = extract_final_response(raw_output)
        candidates.append(
            GeneratedCandidate(
                prompt=candidate_prompt,
                raw_output=raw_output,
                final_response=final_response,
            )
        )
    return candidates
