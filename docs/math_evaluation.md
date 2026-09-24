# MATH-500, AIME 2025, and pass@k evaluation

## Models and configs

Both configs use reasoning mode (`enable_thinking: true`) and these models:

| Model | Hugging Face checkpoint | Subfolder |
| --- | --- | --- |
| Base | `Qwen/Qwen3.5-2B` | None |
| Logit KD | `Khaquan/qwen-khaquanS-distillations` | `qwen-logitKD-final/qwen_logit_kd_continue/epoch_1` |
| ReasonDistill | `Khaquan/qwen-khaquanS-distillations` | `qwen_reasondistill_final_continue/epoch_1` |

`configs/test_math.yaml` evaluates all 500 MATH-500 questions and all 30 AIME 2025
questions at pass@1. `configs/sweep_pass_k.yaml` evaluates the same three models on
MMLU, GSM8K, HellaSwag, AIME 2025, MATH-500 and ARC-C, estimating pass@2, pass@4,
and pass@8 from eight sampled completions per question. The four existing
benchmarks retain the September limit of 1,000 examples each; MATH-500 and AIME
use their complete test sets. HellaSwag uses validation; MMLU uses all subjects.
Benchmark identifiers remain `math500` and `aime25`; the config parser also
accepts `math-500` and `aime-25` aliases.

Generation uses temperature 1.0, top-p 0.95, top-k 20 and repetition penalty 1.0.
MATH-500 and the existing benchmarks receive 4,096 new tokens. AIME receives
16,384. These are maximum generation budgets, not promised response lengths.
No stop-after-box rule is added; EOS and the token budget retain their existing
meaning. Token-limit rates are reported so capped reasoning can be inspected.

The question batch is 1 for every benchmark, matching the training microbatch
in `configs/qwen_logit_kd_continue.yaml`. Gradient accumulation is not used in
evaluation. In the sweep,
`sample_batch_size: 1` generates one sample per question per call, making eight
calls per question batch. This avoids multiplying GPU memory by eight. Increase
these YAML values if the instance has sufficient memory. Batch size, sampling
chunk size, package versions and hardware can change sampled outputs.

## Data and prompting

- MATH-500 loads only `HuggingFaceH4/MATH-500`, `test`, with exactly 500 rows.
  It uses the `answer` field (not the worked `solution`), stable `unique_id`,
  subject and difficulty level.
- AIME 2025 loads only `math-ai/aime25`, `test`, with exactly 30 rows spanning
  AIME I and II. Problem IDs and zero-valued answers are preserved.
- Loading errors, wrong splits/counts, invalid rows and duplicate IDs raise errors.
  These math loaders never substitute a different MATH dataset or a toy example.
  The new configs also reject embedded fallbacks from the other benchmark loaders.
- Prompts use Qwen's official chat template and the step-by-step, final-boxed-answer
  instruction used by Qwen math evaluation. AIME additionally asks for a single
  integer from 0 to 999. The redundant `\boxed{answer}` placeholder is removed
  from these two prompts. Existing GSM8K and multiple-choice prompts are unchanged.

## Extraction and scoring

Only generated completion tokens are decoded. For MATH-500/AIME in Qwen reasoning
mode, a completed `</think>` boundary is required before the final answer. A
completion cut off during reasoning receives no answer credit. We take the last
balanced `\boxed{...}` or `\fbox{...}` in the final response, supporting nested
fractions, radicals, tuples and escaped set braces. A missing, empty or incomplete
last box fails extraction; we do not recover a convenient earlier number.

MATH-500 compares the extracted answer with the gold using **Math-Verify 0.8.0**,
with gold first, strict symbol matching, and five-second parsing/comparison
timeouts. Answers are parsed as whole LaTeX expressions rather than extracting
the first numeric substring. Equivalent fractions, radicals, algebraic
expressions and sets are accepted; different denominators, interval endpoints,
boundary types and variables are distinguished. Identical literal answers also
match. Empty or unparseable unequal answers score false.

AIME uses exact integer equality in [0, 999], accepting leading zeros (`007 = 7`).
Non-integer strings, fractions, prose and lists of possible answers are rejected.
This is an intentionally explicit **final-box protocol**: an unboxed answer does
not receive credit even if its value appears elsewhere in the completion. It may
produce different scores from harnesses that search all reasoning for any answer.

## Pass@k semantics

For a question with `c` correct completions among `n` samples, we calculate:

```text
pass@k = 1 - C(n-c, k) / C(n, k)
```

The reported metric is the mean over questions, following the HumanEval
estimator. All requested k values reuse the same n samples. With n=8, pass@8 is
whether any sample succeeded; pass@2 and pass@4 average over possible subsets,
not just the first two/four samples. This is oracle success probability, not
majority-vote accuracy. On multiple-choice benchmarks it can rise substantially
from chance successes, so do not interpret it as single-answer accuracy.

`generation.pass_at_k` accepts either the previous positive integer or a list.
`num_samples` defaults to the largest requested k, and must be at least that
large. `sample_batch_size` controls generation chunking, independently of k.
Multiple samples require positive temperature. To reduce estimator variance,
increase `num_samples` (for example to 32) without changing `[2, 4, 8]`; that also
increases inference cost. The supplied sweep produces 108,720 completions with
its default sample limits, so a small smoke run is useful first.

## Run on Vast.ai

From the repository root, with the existing CUDA environment activated:

```bash
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v

# Small checks of all three checkpoints; separate outputs:
python -m custom_eval.run_eval --config configs/test_math.yaml \
  --max-samples 2 --output-dir eval_outputs/test_math_smoke
python -m custom_eval.run_eval --config configs/sweep_pass_k.yaml \
  --max-samples 2 --output-dir eval_outputs/sweep_pass_k_smoke

# Full runs, ideally inside tmux:
python -m custom_eval.run_eval --config configs/test_math.yaml \
  --output-dir "eval_outputs/test_math_$(date -u +%Y%m%dT%H%M%SZ)"
python -m custom_eval.run_eval --config configs/sweep_pass_k.yaml \
  --output-dir "eval_outputs/sweep_pass_k_$(date -u +%Y%m%dT%H%M%SZ)"
```

Use a fresh output directory for each run; explicit existing paths can overwrite
results. The math test produces six result JSONs; the sweep produces eighteen,
each containing all requested k estimates. Both write `index.json` and
`run_config.json`. There is no need to load an SAE or aligner for evaluation.

Existing result fields remain. New fields include:

- `summary.pass_at_k_scores`: mapping such as `{"2": ..., "4": ..., "8": ...}`.
- `summary.score` / `score_k`: the estimate at the largest requested k.
- `summary.correct` and record `passed`: count/flag for at least one correct
  answer among all n samples, retained for compatibility. When n exceeds the
  largest k, this count is not the numerator of `summary.score`.
- Per-question `num_samples`, `num_correct`, and `pass_at_k_scores`.
- Candidate `generated_tokens` (excluding EOS and trailing batch padding) and
  `hit_token_limit`; aggregate token-limit and empty-answer rates.
- Checkpoint subfolder, seed, scorer, and Math-Verify version. `run_config.json`
  saves the resolved configuration and core package versions; example metadata
  records math dataset sources and fingerprints.

The runner now resets seed 42 before each model/benchmark generation and frees
model memory between models. This improves repeatability but changes the old
unseeded sampling behavior. Results are not guaranteed bit-identical across
hardware, dependency versions or batch layouts. The existing GSM8K and
multiple-choice scorers are retained for continuity with September.

## Research references and validation

- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math): boxed-answer prompting
  and math evaluation conventions.
- [Math-Verify](https://github.com/huggingface/Math-Verify): mathematical parsing
  and symbolic equivalence; used as a pinned dependency, not copied code.
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and
  [AIME 2025](https://huggingface.co/datasets/math-ai/aime25): fixed benchmark sources.
- [HumanEval pass@k](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py):
  the shared-sample combinatorial estimator.

Local verification covers nested-box extraction, reasoning boundaries, symbolic
comparisons and false positives, zero AIME answers, strict dataset loading,
configuration validation, pass@k boundaries, and a mocked end-to-end runner.
The actual public datasets were loaded and all 500 MATH-500 gold answers parsed.
Full GPU model evaluations must still be run on the instance.
