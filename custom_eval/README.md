# Custom Evaluation Harness

Run benchmark evaluations from YAML against either Hugging Face model ids or local `save_pretrained` checkpoint directories produced by this repo.

```bash
python custom_eval/run_eval.py --config examples/eval_harness_example.yaml
```

The harness supports `arc-c`, `math500`, `aime25`, `gsm8k`, `hellaswag`, `mmlu`, and `gpqa`.

## Required YAML shape

```yaml
models:
  - name: my-local-model
    checkpoint: ./checkpoints/amdeepseek_hard_label/epoch_3

benchmarks: [gsm8k, math500]

generation:
  cot: false
  pass_at_k: 1
  max_new_tokens: 128
```

For chain-of-thought style runs, `max_toks` is required and is used as the generation budget:

```yaml
generation:
  cot: true
  max_toks: 512
  pass_at_k: 4
  temperature: 0.7
```

Every prompt instructs the model to put the final answer on its own line as:

```text
final response: {answer}
```

If the model omits that marker, the harness falls back to scraping the last generated line.

## Outputs

Each model/benchmark pair is saved as:

```text
eval_outputs/{model_name}__{benchmark}.json
```

The JSON contains:

- `summary`: score, pass@k setting, model checkpoint, benchmark, timing, and CoT settings.
- `records`: one entry per problem, including the prompt, raw generations, scraped final responses, correctness flags, and benchmark metadata.

`eval_outputs/index.json` collects all output paths for the run.

## Benchmark loading and fallbacks

Each benchmark adapter first tries likely Hugging Face dataset ids and split names. If those are unavailable, offline, or blocked, it falls back to a tiny embedded smoke-test example so the harness still runs end-to-end and exposes dataset loading errors in each example's metadata.

