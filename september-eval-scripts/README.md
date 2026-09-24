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

For full benchmark splits, set `max_samples: null` in both YAML files. The default 1,000-example limit and sampling parameters match the earlier runs. All benchmarks receive 4,096 new tokens in both modes; ARC-C and HellaSwag previously used 1,024. Inspect thinking outputs for truncation before interpreting scores. `seed: 42` is retained. The runner now seeds each model/benchmark generation; historical September runs predate this change. Hardware, package versions, and batch layouts can still affect reproducibility.

## Logit KD versus ReasonDistill, reasoning only

This comparison runs exactly two checkpoints from
`Khaquan/qwen-khaquanS-distillations`:

- `qwen-logitKD-final/qwen_logit_kd/epoch_1`
- `qwen_reasondistill_final/epoch_1`

Both use `enable_thinking: true`. The four benchmarks, dataset splits, sample
limit, prompts, scoring, model-loading options, and generation settings match
the September reasoning runs: GSM8K, MMLU (all subjects), ARC-Challenge, and
HellaSwag (validation), up to 1,000 examples each, 4,096 new tokens, batch size 32,
temperature 1.0, top-p 0.95, top-k 20, repetition penalty 1.0, and pass@1.
The original benchmark order is retained. Neither a base model nor either
continuation checkpoint is included. This produces eight result JSON files
plus an index.

Run on the CUDA instance using the existing evaluation environment:

```bash
# Optional small check, saved separately from the full run:
bash september-eval-scripts/run_kd_comparison.sh --max-samples 2

# Full comparison with the September sample limit:
bash september-eval-scripts/run_kd_comparison.sh
```

Each invocation writes a fresh directory under
`eval_outputs/september-kd-comparison/`, with results in its `think/` subdirectory.
`PYTHON` and `EVAL_OUTPUT_ROOT` work as in the original launcher. It uses the shared evaluation
runner, with the updated seeding behavior and dataset-fallback
limitations described above. For the small check, confirm that examples come
from the real datasets and that checkpoint loading reports no unintended
missing weights before starting the full run.

## SAE and ReasonScore

These artifacts are used during training, not loaded by `custom_eval` for checkpoint evaluation. The existing training configuration remains unchanged:

- SAE: `hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/sae.pt`
- ReasonScore: `hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/reasonscore.pt`

The distilled model keeps the existing `strip_language_model_prefix: true` setting. Check model-loading logs for missing/unexpected weights. In the smoke results, verify actual dataset sources (the harness can fall back to embedded examples), clean completion boundaries, different thinking-mode prompts, and sensible extracted answers before a full run.


## Average generation length on 500 GSM8K questions

Generate fresh responses from exactly the same first 500 `openai/gsm8k` main/test questions using the two September checkpoints:

```bash
# Choose one mode for both models:
python september-eval-scripts/gsm8k_generation_length.py --mode think
# Or:
python september-eval-scripts/gsm8k_generation_length.py --mode no_think
```

Use the working CUDA evaluation environment. Defaults match September: batch size 32, 4,096 new tokens, temperature 1.0, top-p 0.95, top-k 20, and repetition penalty 1.0. This script additionally resets generation seed 42 for each model. Reproducibility still depends on package/kernel versions and batch size. The dataset is loaded once without embedded fallback, and both models receive identical questions in the same order. Each model is released before loading the next.

Generation length counts actual generated token IDs, excluding the padded input, terminal EOS token, and padding after EOS. It includes both reasoning and final-answer tokens, plus any generated internal special tokens. The script also records generation steps including EOS, and checks matching tokenizer vocabularies and formatted prompts before producing the paired comparison.

The 4,096-token budget can censor long outputs. The mean includes these budget-limited responses, and the summary reports their frequency separately. EOS-terminated-only means describe a selected subset and should not replace the main comparison. Increase the cap for both models with `--max-new-tokens 8192` if desired; use `--batch-size 8` to reduce memory demand. A two-item check is available via `--num-samples 2`.

Results go to a new timestamped `eval_outputs/gsm8k-length-MODE-*` directory:

- `summary.json`: each model's mean, median, sample standard deviation, p90/p95, min/max, total token count, EOS count, token-limit rate, and paired mean difference/ratio.
- `paired_lengths.csv`: one row per shared question with both lengths and their difference.
- `base.jsonl`, `distilled.jsonl`: raw responses, exact token lengths, prompts, stop reasons, and repetition/closing-tag diagnostics; flushed after each batch.
- `metadata.json`, `questions.json`: selected questions, dataset fingerprint, package versions, hardware, generation settings, and question hash.
- `base-summary.json`, `distilled-summary.json`: individual summaries, retained if a later model fails.

Pass `--output-dir PATH` to select a new output directory. Existing directories are rejected to prevent accidental overwrites. No benchmark accuracy is computed by this script.


### Stop after the first completed final-answer format

```bash
python september-eval-scripts/gsm8k_generation_length.py \
  --mode think --stop-at-final-answer
```

This optional flag stops each sequence at the first nonempty `<answer>...</answer>` block or balanced `\boxed{...}` outside a thinking block. In thinking mode it waits for `</think>` first; in either mode it also ignores answers inside any newly emitted `<think>` block. A box inside an open answer block waits for `</answer>`. Empty boxes and the literal `answer` placeholder are ignored. Plain answers without either recognized format still terminate on EOS or the token budget.

The callback returns one stop decision per batch member. Completed members are padded by Transformers while other members continue, and that padding is excluded from measured lengths. Counts include the token completing the answer marker (which may also contain a few trailing characters). The script saves `finish_reason: final_answer`, the stop kind (`answer_tag` or `boxed`), and `num_final_answer_terminated` in its summaries. It does not inject an EOS token.

With this flag, the metric is **tokens until the first recognized completed answer**, not the full natural output length. Apply the same flag to both models and keep these results separate from older unrestricted runs. Format completion does not guarantee correctness or prevent a model from wanting to revise its answer. The existing benchmark runner is unchanged. Custom text checks add some stopping overhead; a batch still runs until its slowest member finishes.
