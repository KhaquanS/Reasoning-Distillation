"""
Main evaluation runner with batched inference and pass@k support.
"""

import gc
import json
from dataclasses import asdict
from importlib.metadata import version
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import set_seed
from custom_eval.metrics import estimate_pass_at_k

from custom_eval.benchmarks import load_benchmark
from custom_eval.config import EvalConfig
from custom_eval.generation import generate_candidates_batch
from custom_eval.modeling import load_model_and_tokenizer


def _slug(text: str) -> str:
    """Create a safe filename from text."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def _get_benchmark_param(
    config: EvalConfig,
    benchmark_name: str,
    param_name: str,
    default: Any,
) -> Any:
    """
    Get a parameter from benchmark_options if present, else use global config.
    """
    return config.benchmark_options.get(benchmark_name, {}).get(param_name, default)


def run_evaluation(config: EvalConfig) -> List[Dict[str, Any]]:
    """
    Run the full evaluation with batched example processing.
    """
    # Show CUDA status at start
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = config.pass_at_k if isinstance(config.pass_at_k, list) else [config.pass_at_k]
    num_samples = config.num_samples or max(k_values)
    sample_batch_size = config.sample_batch_size or num_samples
    if num_samples < max(k_values) or not 1 <= sample_batch_size <= num_samples:
        raise ValueError("Invalid sample count or sample_batch_size.")
    with (output_dir / "run_config.json").open("w") as f:
        json.dump({"config": asdict(config), "versions": {
            name: version(name) for name in ("torch", "transformers", "datasets")
        }}, f, indent=2)

    all_results = []

    for model_spec in config.models:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_spec.name} ({model_spec.checkpoint})")
        print(f"Model type: {model_spec.model_type}")
        print(f"Thinking mode: {'Enabled' if model_spec.enable_thinking else 'Disabled'}")
        print(f"Batch size: {config.batch_size}")
        print(f"Pass@k: {config.pass_at_k}")
        print(f"{'='*60}")

        # Load the model
        model, tokenizer = load_model_and_tokenizer(model_spec, config.cache_dir)

        print(f"Model device: {model.device}")
        if hasattr(model, "hf_device_map"):
            print(f"Device map: {model.hf_device_map}")
        print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
        print()

        model_results = []

        for benchmark_name in config.benchmarks:
            print(f"\n--- Running {benchmark_name} ---")

            # Get effective parameters for this benchmark
            effective_max_tokens = _get_benchmark_param(
                config, benchmark_name, "max_new_tokens", config.max_new_tokens
            )
            effective_temp = _get_benchmark_param(
                config, benchmark_name, "temperature", config.temperature
            )
            effective_top_p = _get_benchmark_param(
                config, benchmark_name, "top_p", config.top_p
            )
            effective_top_k = _get_benchmark_param(
                config, benchmark_name, "top_k", config.top_k
            )
            effective_repetition_penalty = _get_benchmark_param(
                config, benchmark_name, "repetition_penalty", config.repetition_penalty
            )

            if num_samples > 1 and effective_temp <= 0:
                raise ValueError("Pass@k sampling requires temperature > 0 for every benchmark.")
            # Reset for each model/benchmark so comparisons use the same RNG seed.
            set_seed(config.seed)
            print(f"Effective max_new_tokens: {effective_max_tokens}")

            # Load benchmark data
            benchmark = load_benchmark(
                benchmark_name,
                cache_dir=config.cache_dir,
                split=config.split,
                max_samples=config.max_samples,
                seed=config.seed,
                options=config.benchmark_options.get(benchmark_name, {}),
            )

            examples = benchmark.examples
            if not examples:
                raise ValueError(f"No examples loaded for {benchmark_name}.")
            if config.require_real_data and any(
                (ex.metadata or {}).get("source") == "embedded_fallback" for ex in examples
            ):
                raise RuntimeError(f"{benchmark_name} dataset failed to load; refusing embedded fallback.")
            total = len(examples)
            print(f"Examples: {total}")

            records = []
            correct_count = 0
            pass_sums = {k: 0.0 for k in k_values}
            token_limit_count = 0
            empty_answer_count = 0
            started = time.time()

            batch_size = int(_get_benchmark_param(config, benchmark_name, "batch_size", config.batch_size))
            if batch_size <= 0:
                raise ValueError("Benchmark batch_size must be positive.")
            for i in tqdm(range(0, total, batch_size), desc=f"{model_spec.name}/{benchmark_name}"):
                batch_examples = examples[i:i+batch_size]
                questions = [ex.question for ex in batch_examples]

                candidates_per_question = [[] for _ in questions]
                for offset in range(0, num_samples, sample_batch_size):
                    count = min(sample_batch_size, num_samples - offset)
                    chunk = generate_candidates_batch(
                        model=model, tokenizer=tokenizer, questions=questions,
                        benchmark_name=benchmark_name, model_type=model_spec.model_type,
                        enable_thinking=model_spec.enable_thinking,
                        max_new_tokens=effective_max_tokens, temperature=effective_temp,
                        top_p=effective_top_p, top_k=effective_top_k,
                        repetition_penalty=effective_repetition_penalty,
                        pass_at_k=count, system_prompt=None, max_input_length=4096,
                        show_progress=True,
                    )
                    if len(chunk) != len(questions) or any(len(group) != count for group in chunk):
                        raise RuntimeError("Generation returned an unexpected number of candidates.")
                    for collected, group in zip(candidates_per_question, chunk):
                        collected.extend(group)

                # Process each example in the batch
                for ex, candidates in zip(batch_examples, candidates_per_question):
                    candidate_records = []
                    passed = False
                    for idx, cand in enumerate(candidates):
                        is_correct = bool(benchmark.scorer(cand.final_response, ex.answer))
                        token_limit_count += int(bool(cand.hit_token_limit))
                        empty_answer_count += int(not cand.final_response)
                        passed = passed or is_correct
                        candidate_records.append({
                            "index": idx,
                            "prompt": cand.prompt,
                            "raw_output": cand.raw_output,
                            "final_response": cand.final_response,
                            "thinking_content": cand.thinking_content,
                            "correct": is_correct,
                            "generated_tokens": cand.generated_tokens,
                            "hit_token_limit": cand.hit_token_limit,
                        })
                    correct_count += int(passed)
                    num_correct = sum(c["correct"] for c in candidate_records)
                    estimates = {str(k): estimate_pass_at_k(num_samples, num_correct, k) for k in k_values}
                    for k in k_values:
                        pass_sums[k] += estimates[str(k)]
                    records.append({
                        "id": ex.id,
                        "question": ex.question,
                        "answer": ex.answer,
                        "metadata": ex.metadata,
                        "passed": passed,
                        "num_samples": num_samples,
                        "num_correct": num_correct,
                        "pass_at_k_scores": estimates,
                        "candidates": candidate_records,
                    })

            elapsed = time.time() - started
            summary = {
                "model": model_spec.name,
                "checkpoint": model_spec.checkpoint,
                "benchmark": benchmark.name,
                "num_examples": total,
                "score": pass_sums[max(k_values)] / total,
                "score_k": max(k_values),
                "pass_at_k_scores": {str(k): pass_sums[k] / total for k in k_values},
                "estimator": "1 - C(n-c,k) / C(n,k)",
                "num_samples_per_question": num_samples,
                "sample_batch_size": sample_batch_size,
                "token_limit_rate": token_limit_count / (total * num_samples),
                "empty_answer_rate": empty_answer_count / (total * num_samples),
                "seed": config.seed,
                "subfolder": model_spec.subfolder,
                "scorer": benchmark.scorer.__name__,
                "math_verify_version": version("math-verify") if benchmark_name == "math500" else None,
                "correct": correct_count,
                "pass_at_k": config.pass_at_k,
                "enable_thinking": model_spec.enable_thinking,
                "model_type": model_spec.model_type,
                "max_new_tokens": effective_max_tokens,
                "temperature": effective_temp,
                "top_p": effective_top_p,
                "top_k": effective_top_k,
                "repetition_penalty": effective_repetition_penalty,
                "batch_size": batch_size,
                "elapsed_seconds": round(elapsed, 3),
            }

            payload = {"summary": summary, "records": records}
            output_path = output_dir / f"{_slug(model_spec.name)}__{benchmark_name}.json"
            with output_path.open("w") as f:
                json.dump(payload, f, indent=2)

            print(f"Saved: {output_path}")
            print(f"Pass@k: {summary['pass_at_k_scores']}")

            model_results.append({
                "summary": summary,
                "output_path": str(output_path),
            })

        all_results.append({"model": model_spec.name, "benchmarks": model_results})
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_path = output_dir / "index.json"
    with index_path.open("w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Index: {index_path}")
    print(f"{'='*60}")

    return all_results