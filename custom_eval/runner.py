"""
Main evaluation runner.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from custom_eval.benchmarks import load_benchmark
from custom_eval.config import EvalConfig
from custom_eval.generation import generate_candidates
from custom_eval.modeling import load_model_and_tokenizer


def _slug(text: str) -> str:
    """Create a safe filename from text."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def run_evaluation(config: EvalConfig) -> List[Dict[str, Any]]:
    """
    Run the full evaluation.
    
    Args:
        config: Evaluation configuration
    
    Returns:
        List of results for each model/benchmark pair
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for model_spec in config.models:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_spec.name} ({model_spec.checkpoint})")
        print(f"Thinking mode: {'Enabled' if model_spec.enable_thinking else 'Disabled'}")
        print(f"{'='*60}")
        
        model, tokenizer = load_model_and_tokenizer(model_spec, config.cache_dir)
        model_results = []
        
        for benchmark_name in config.benchmarks:
            print(f"\n--- Running {benchmark_name} ---")
            
            # Load benchmark data
            benchmark = load_benchmark(
                benchmark_name,
                cache_dir=config.cache_dir,
                split=config.split,
                max_samples=config.max_samples,
                seed=config.seed,
                options=config.benchmark_options.get(benchmark_name, {}),
            )
            
            print(f"Examples: {len(benchmark.examples)}")
            
            records = []
            correct_count = 0
            started = time.time()
            
            # Process examples
            for example in tqdm(benchmark.examples, desc=f"{model_spec.name}/{benchmark_name}"):
                candidates = generate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    question=example.question,
                    benchmark_name=benchmark_name,
                    enable_thinking=model_spec.enable_thinking,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    presence_penalty=config.presence_penalty,
                    repetition_penalty=config.repetition_penalty,
                    pass_at_k=1,  # Currently only support pass_at_k=1 in generation
                    batch_size=config.batch_size,
                )
                
                # Score candidates
                candidate_records = []
                passed = False
                
                for idx, candidate in enumerate(candidates):
                    is_correct = benchmark.scorer(candidate.final_response, example.answer)
                    passed = passed or is_correct
                    candidate_records.append({
                        "index": idx,
                        "prompt": candidate.prompt,
                        "raw_output": candidate.raw_output,
                        "final_response": candidate.final_response,
                        "thinking_content": candidate.thinking_content,
                        "correct": is_correct,
                    })
                
                correct_count += int(passed)
                records.append({
                    "id": example.id,
                    "question": example.question,
                    "answer": example.answer,
                    "metadata": example.metadata,
                    "passed": passed,
                    "candidates": candidate_records,
                })
            
            # Compute summary
            total = len(benchmark.examples)
            elapsed = time.time() - started
            
            summary = {
                "model": model_spec.name,
                "checkpoint": model_spec.checkpoint,
                "benchmark": benchmark.name,
                "num_examples": total,
                "score": correct_count / total if total else 0.0,
                "correct": correct_count,
                "enable_thinking": model_spec.enable_thinking,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "presence_penalty": config.presence_penalty,
                "repetition_penalty": config.repetition_penalty,
                "elapsed_seconds": round(elapsed, 3),
            }
            
            payload = {"summary": summary, "records": records}
            
            output_path = output_dir / f"{_slug(model_spec.name)}__{benchmark_name}.json"
            with output_path.open("w") as f:
                json.dump(payload, f, indent=2)
            
            print(f"Saved: {output_path}")
            print(f"Score: {summary['score']:.4f} ({summary['correct']}/{total})")
            
            model_results.append({
                "summary": summary,
                "output_path": str(output_path),
            })
        
        all_results.append({"model": model_spec.name, "benchmarks": model_results})
    
    # Save index
    index_path = output_dir / "index.json"
    with index_path.open("w") as f:
        json.dump({"results": all_results}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Index: {index_path}")
    print(f"{'='*60}")
    
    return all_results