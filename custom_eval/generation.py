"""
Generation utilities for Qwen models with proper chat formatting.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        # Find JSON-like structure
        match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try full JSON parse
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
    Extract final response from model output based on benchmark type.
    
    Args:
        text: Raw model output
        benchmark_name: Name of the benchmark
        enable_thinking: Whether thinking mode was enabled
    
    Returns:
        Extracted final answer
    """
    if not text:
        return ""
    
    # First, extract response after thinking if enabled
    if enable_thinking:
        text = QwenChatFormatter.extract_response(text)
    
    text = text.strip()
    
    # Try benchmark-specific extraction
    if benchmark_name in {"math500", "aime25", "gsm8k"}:
        # Math benchmarks: look for boxed answer
        answer = extract_final_answer_from_boxed(text)
        if answer:
            return answer
    
    if benchmark_name in {"arc-c", "mmlu", "gpqa", "hellaswag"}:
        # Multiple choice: look for JSON answer
        answer = extract_json_answer(text)
        if answer:
            return answer
    
    # Fallback: look for common patterns
    patterns = [
        r"final\s+answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
        r"therefore\s*,\s*(.+)$",
        r"so\s*,\s*(.+)$",
        r"the answer is\s*(.+)",
        r"^([A-D])$",  # Single letter answer
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer
    
    # Last resort: return the last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


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
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    pass_at_k: int = 1,
    batch_size: int = 1,
    system_prompt: Optional[str] = None,
) -> List[GeneratedCandidate]:
    """
    Generate candidates for a given question.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question to answer
        benchmark_name: Name of the benchmark
        enable_thinking: Whether to enable thinking mode
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        presence_penalty: Presence penalty
        repetition_penalty: Repetition penalty
        pass_at_k: Number of candidates to generate
        batch_size: Batch size for generation
        system_prompt: Optional system prompt
    
    Returns:
        List of generated candidates
    """
    # Build the prompt using benchmark-specific template
    prompt = build_prompt(question, benchmark_name)
    
    # Format messages for Qwen
    messages = QwenChatFormatter.format_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    candidates = []
    
    for _ in range(pass_at_k):
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        generation_kwargs = QwenChatFormatter.create_generation_kwargs(
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
        
        # Remove any tokenizer-specific kwargs that might cause issues
        generation_kwargs.pop("chat_template_kwargs", None)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = output_ids[0, inputs["input_ids"].shape[-1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # Extract final response
        final_response = extract_final_response(
            raw_output,
            benchmark_name=benchmark_name,
            enable_thinking=enable_thinking,
        )
        
        # Extract thinking content if any
        thinking_content = None
        if enable_thinking:
            think_pattern = re.compile(r"<think\s*>\s*(.*?)\s*</think\s*>", re.IGNORECASE | re.DOTALL)
            match = think_pattern.search(raw_output)
            if match:
                thinking_content = match.group(1).strip()
        
        candidates.append(
            GeneratedCandidate(
                prompt=prompt,
                raw_output=raw_output,
                final_response=final_response,
                thinking_content=thinking_content,
            )
        )
    
    return candidates