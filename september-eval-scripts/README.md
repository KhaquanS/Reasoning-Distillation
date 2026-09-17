# September evaluations

Two models × two modes × four benchmarks = 16 model/mode/benchmark combinations.

- Distilled: `Khaquan/qwen-khaquanS-distillations`, subfolder `qwen_reasondistill_final_continue/epoch_1`.
- Base: `Qwen/Qwen3.5-2B`, matching the previous base runs.
- Modes: `enable_thinking: true` and `false`.
- Benchmarks: GSM8K, MMLU (all subjects), ARC-Challenge, HellaSwag (validation).

Use the repository's evaluation environment with its dependencies installed and sufficient GPU memory. The configs use bfloat16 and batch size 32; lower the batch size in both configs if necessary. No evaluation is launched by creating these files.

## Run

From the repository root:

```bash
# Small end-to-end check of both models and modes on all four benchmarks:
bash september-eval-scripts/run.sh all --max-samples 2

# Standard run: up to 1,000 examples per benchmark:
bash september-eval-scripts/run.sh all

# Run just one mode:
bash september-eval-scripts/run.sh think
bash september-eval-scripts/run.sh no_think
```

Set `PYTHON=/path/to/python` to select an interpreter. Each launcher invocation writes to a fresh directory under `eval_outputs/september/`, with separate `think` and `no_think` directories and indexes. Set `EVAL_OUTPUT_ROOT` to choose that parent directory. Avoid reusing it if you want to retain earlier results. The launcher accepts the eval CLI's additional options; an explicit `--output-dir` overrides mode separation, so prefer `EVAL_OUTPUT_ROOT`.

For full benchmark splits, set `max_samples: null` in both YAML files. The default 1,000-example limit and sampling parameters match the earlier runs. All benchmarks receive 4,096 new tokens in both modes; ARC-C and HellaSwag previously used 1,024. Inspect thinking outputs for truncation before interpreting scores. `seed: 42` is retained, but the current runner does not seed model sampling, so it does not guarantee identical generated outputs between runs.

## SAE and ReasonScore

These artifacts are used during training, not loaded by `custom_eval` for checkpoint evaluation. The existing training configuration remains unchanged:

- SAE: `hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/sae.pt`
- ReasonScore: `hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/reasonscore.pt`

The distilled model keeps the existing `strip_language_model_prefix: true` setting. Check model-loading logs for missing/unexpected weights. In the smoke results, verify actual dataset sources (the harness can fall back to embedded examples), clean completion boundaries, different thinking-mode prompts, and sensible extracted answers before a full run.
