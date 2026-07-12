"""
Main evaluation runner with batched inference and pass@k support.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from custom_eval.benchmarks import load_benchmark
from custom_eval.config import EvalConfig
from custom_eval.generation import generate_candidates_batch
from custom_eval.modeling import load_model_and_tokenizer


def _slug(text: str) -> str:
    """Create a safe filename from text."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def run_evaluation(config: EvalConfig) -> List[Dict[str, Any]]:
    """
    Run the full evaluation with batched example processing.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_spec in config.models:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_spec.name} ({model_spec.checkpoint})")
        print(f"Thinking mode: {'Enabled' if model_spec.enable_thinking else 'Disabled'}")
        print(f"Batch size: {config.batch_size}")
        print(f"Pass@k: {config.pass_at_k}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_spec, config.cache_dir)
        model_results = []

        for benchmark_name in config.benchmarks:
            print(f"\n--- Running {benchmark_name} ---")

            benchmark = load_benchmark(
                benchmark_name,
                cache_dir=config.cache_dir,
                split=config.split,
                max_samples=config.max_samples,
                seed=config.seed,
                options=config.benchmark_options.get(benchmark_name, {}),
            )

            examples = benchmark.examples
            total = len(examples)
            print(f"Examples: {total}")

            records = []
            correct_count = 0
            started = time.time()

            batch_size = config.batch_size
            for i in tqdm(range(0, total, batch_size), desc=f"{model_spec.name}/{benchmark_name}"):
                batch_examples = examples[i:i+batch_size]
                questions = [ex.question for ex in batch_examples]

                # Generate candidates for the whole batch
                candidates_per_question = generate_candidates_batch(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    benchmark_name=benchmark_name,
                    enable_thinking=model_spec.enable_thinking,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    pass_at_k=config.pass_at_k,
                    system_prompt=None,
                    max_input_length=4096,
                )

                # Process each example in the batch
                for ex, candidates in zip(batch_examples, candidates_per_question):
                    candidate_records = []
                    passed = False
                    for idx, cand in enumerate(candidates):
                        is_correct = benchmark.scorer(cand.final_response, ex.answer)
                        passed = passed or is_correct
                        candidate_records.append({
                            "index": idx,
                            "prompt": cand.prompt,
                            "raw_output": cand.raw_output,
                            "final_response": cand.final_response,
                            "thinking_content": cand.thinking_content,
                            "correct": is_correct,
                        })
                    correct_count += int(passed)
                    records.append({
                        "id": ex.id,
                        "question": ex.question,
                        "answer": ex.answer,
                        "metadata": ex.metadata,
                        "passed": passed,
                        "candidates": candidate_records,
                    })

            elapsed = time.time() - started
            summary = {
                "model": model_spec.name,
                "checkpoint": model_spec.checkpoint,
                "benchmark": benchmark.name,
                "num_examples": total,
                "score": correct_count / total if total else 0.0,
                "correct": correct_count,
                "pass_at_k": config.pass_at_k,
                "enable_thinking": model_spec.enable_thinking,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "batch_size": config.batch_size,
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

    index_path = output_dir / "index.json"
    with index_path.open("w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Index: {index_path}")
    print(f"{'='*60}")

    return all_results