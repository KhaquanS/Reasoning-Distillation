import json
import time
from pathlib import Path

from tqdm import tqdm

from custom_eval.benchmarks import load_benchmark
from custom_eval.config import EvalConfig
from custom_eval.generation import generate_candidates
from custom_eval.modeling import load_model_and_tokenizer


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def run_evaluation(config: EvalConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_spec in config.models:
        print(f"Loading model: {model_spec.name} ({model_spec.checkpoint})")
        model, tokenizer = load_model_and_tokenizer(model_spec, config.cache_dir)
        model_results = []

        for benchmark_name in config.benchmarks:
            benchmark = load_benchmark(
                benchmark_name,
                cache_dir=config.cache_dir,
                split=config.split,
                max_samples=config.max_samples,
                seed=config.seed,
                options=config.benchmark_options.get(benchmark_name, {}),
            )
            print(f"Running {benchmark.name}: {len(benchmark.examples)} examples")

            records = []
            correct_count = 0
            started = time.time()
            for example in tqdm(benchmark.examples, desc=f"{model_spec.name}/{benchmark.name}"):
                candidates = generate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    question=example.question,
                    cot=config.cot,
                    max_toks=config.max_toks,
                    pass_at_k=config.pass_at_k,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                candidate_records = []
                passed = False
                for index, candidate in enumerate(candidates):
                    is_correct = benchmark.scorer(candidate.final_response, example.answer)
                    passed = passed or is_correct
                    candidate_records.append(
                        {
                            "index": index,
                            "prompt": candidate.prompt,
                            "raw_output": candidate.raw_output,
                            "final_response": candidate.final_response,
                            "correct": is_correct,
                        }
                    )
                correct_count += int(passed)
                records.append(
                    {
                        "id": example.id,
                        "question": example.question,
                        "answer": example.answer,
                        "metadata": example.metadata,
                        "passed": passed,
                        "candidates": candidate_records,
                    }
                )

            total = len(benchmark.examples)
            summary = {
                "model": model_spec.name,
                "checkpoint": model_spec.checkpoint,
                "benchmark": benchmark.name,
                "num_examples": total,
                "pass_at_k": config.pass_at_k,
                "score": correct_count / total if total else 0.0,
                "correct": correct_count,
                "cot": config.cot,
                "max_toks": config.max_toks,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            payload = {"summary": summary, "records": records}
            output_path = output_dir / f"{_slug(model_spec.name)}__{benchmark.name}.json"
            with output_path.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved {output_path}")
            model_results.append({"summary": summary, "output_path": str(output_path)})

        all_results.append({"model": model_spec.name, "benchmarks": model_results})

    index_path = output_dir / "index.json"
    with index_path.open("w") as f:
        json.dump({"results": all_results}, f, indent=2)
    return all_results

