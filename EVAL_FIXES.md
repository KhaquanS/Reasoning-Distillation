# Eval fixes (`eval-fixes` branch)

This branch starts from `awesomeReasonDistill`. Training code is unchanged. The differences are fixes to the evaluation harness in `custom_eval/` and the token budget in the thinking eval config.

**Results produced before these fixes don't measure the models and shouldn't be compared.** See [Evidence](#evidence-from-the-earlier-runs) for what was wrong with them.

| # | Fix | Files |
|---|-----|-------|
| 1 | Prompt text leaking into model outputs | `generation.py` |
| 2 | `enable_thinking` now actually reaches the chat template | `generation.py` |
| 3 | Thinking handled correctly when parsing | `parsing.py`, `generation.py` |
| 4 | One parser for every answer format, including `<answer>` tags | `parsing.py` (new), `generation.py`, `scoring.py` |
| 5 | Strict multiple-choice letters | `parsing.py`, `scoring.py` |
| 6 | Math uses the last number, not the first | `parsing.py` |
| 7 | Each output records which parsing rule was used | `generation.py`, `runner.py` |
| 8 | Bigger token budget for thinking runs | `final-run-evals/qwen_2B-think-new.yaml` |

Code files are in `custom_eval/`.

---

## Evidence from the earlier runs

These fixes were checked against the stored results in `eval_jsons/` (31 files, 1000 questions each). Each file stores the prompt, the raw output and the parsed answer for every question.

- **Prompt leak (Fix 1).** In every batch, every output except the longest prompt's starts with the end of its own prompt. That's 991–993 of 1000 rows at batch size 128, and 965–968 at 32. Re-tokenizing the prompts predicted the leaked text exactly for 6000 of 6000 rows checked. The parser read the leaked example answer, so no-think runs scored `C` on 93–99% of multiple-choice rows and `answer` on 93–98% of gsm8k rows.
- **Thinking never on (Fix 2).** Every prompt in the "thinking" runs ends with the empty `<think>\n\n</think>` block, which tells Qwen not to think. For the base model and for reasondistill-new-epoch_1, the think and no-think prompts are identical for 1000/1000 questions on every benchmark. The base model's outputs contain no reasoning in either run.
- **Why "thinking" runs scored higher.** The stored no-think outputs were re-scored with the old parser, flipping only `enable_thinking`. gsm8k went from 1.4% to 78.8%, which is the "thinking" score. With the flag on, the old parser cut at the first `<think></think>` block, and that block was part of the leaked prompt, so it accidentally removed the leak.
- **`<answer>` tags ignored (Fix 4).** Distilled models write `<answer>…</answer>` in 48–97% of outputs depending on the checkpoint (their training format). The old parser couldn't read it, so even their "thinking" scores were undercounted.

### Scores from the stored outputs, re-parsed
The leaked prompt text was removed from each stored output, and the output was parsed with this branch's parser. **Thinking was off in all of these runs**, whatever the run was called. Scores are from a single sample at temperature 1.0.

| Model (no-think run / "think" run) | arc-c | gsm8k | hellaswag | mmlu |
|---|---|---|---|---|
| qwen-3.5-2b (base) | .737 / .764 | .791 / .777 | .398 / .441 | .560 / .536 |
| reasondistill-new-epoch_1 | .628 / .634 | .716 / .727 | .342 / .362 | .499 / .505 |
| reasondistill-continue-epoch_1 | .675 / – | .762 / .759 | .358 / – | .542 / .515 |
| reasondistill-step_60 (no-think only) | .657 | .662 | .336 | .502 |
| reasondistill-step_80 (no-think only) | .638 | .660 | .359 | .517 |

For comparison, the stored scores were ~.25–.28 on multiple choice and ~.01–.03 on gsm8k for every no-think run.

---

## Fix 1: prompt text leaking into model outputs

### The problem
Prompts are batched and **left-padded**: shorter prompts get padding at the front so every row has the same length.

```
[ padding | prompt | generated answer ]
|<---- padded width ---->|
```

The old code decided where each answer starts from that prompt's **unpadded** length. For every prompt shorter than the longest in its batch, that position falls inside the prompt, so the end of the prompt was decoded as if the model had written it. That text includes the example answer from our instructions (`{"answer": "C"}` or `\boxed{answer}`), and the parser read it instead of the model's answer.

### The fix
Every row's answer now starts at the padded width, which is where `model.generate` puts the first new token when inputs are left-padded.

```python
# before
gen_tokens = out[orig_lengths[i]:]      # orig_lengths = attention_mask.sum(dim=1)
# after
gen_tokens = out[input_ids.shape[1]:]
```

---

## Fix 2: `enable_thinking` reaches the chat template

### The problem
`enable_thinking` went from the YAML into `format_messages()`, which accepted it and dropped it. It was never passed to `apply_chat_template`, so Qwen's template used its default, which is **thinking off**. Every run was a non-thinking run, whatever the config said.

This was confirmed with the real Qwen3.5-2B:
- The prompt our code built was identical to Qwen's official thinking-off prompt.
- On the same question, our prompt gave an 87-character direct answer. A real thinking prompt gave 381 characters of reasoning, then `</think>`, then the answer.

### The fix
`enable_thinking` is passed to `apply_chat_template`. Prompts now end with:

| Setting | Prompt ends with |
|---|---|
| `enable_thinking: true` | `<|im_start|>assistant\n<think>\n` |
| `enable_thinking: false` | `<|im_start|>assistant\n<think>\n\n</think>\n\n` |

Templates that don't use the variable ignore it.

**Expect thinking runs to change a lot:** outputs will be much longer, runs slower, and scores different from any earlier "thinking" number.

---

## Fix 3: thinking handled correctly when parsing

In thinking mode the opening `<think>` is part of the **prompt**, so the model's output contains only the closing `</think>`. The old code only removed reasoning when both tags were in the output.

1. **Reasoning is ignored.** Everything up to the last `</think>` is dropped before parsing and saved as `thinking_content`. Draft answers inside the reasoning (e.g. an early `\boxed{12}`) are no longer picked up. This applies in every mode.
2. **Cut-off reasoning counts as no answer.** If the prompt opened a `<think>` block and the output never closes it, the model ran out of tokens mid-reasoning. The answer is `""` and the parse rule is `unfinished_thinking`. This is decided from the actual prompt, not the YAML setting, so it can never apply to a prompt that didn't open a think block.
3. **Answer written inside the reasoning.** If the output ends right at `</think>` with nothing after it, the text before `</think>` is parsed instead.

---

## Fix 4: one parser for every answer format

### The problem
Parsing was split between `generation.py` and `scoring.py`, and it:
- didn't understand `<answer>…</answer>`, the distilled models' training format
- took the **first** `\boxed{}` and broke on nested braces (`\frac{1}{2}`)
- used loose fallbacks (`so, (.+)`, `therefore, (.+)`) that grabbed whole paragraphs

### The fix
All extraction lives in `custom_eval/parsing.py` (`extract_answer`), and the same rules run for every benchmark. The rules are tried in order. A rule only counts if it gives a **usable** answer: a choice label for multiple choice, or a number for gsm8k/aime25. Otherwise the next rule is tried.

| Order | Rule | What it reads |
|---|------|---------------|
| – | `unfinished_thinking` | thinking was opened but never closed (see Fix 3): no answer |
| 1 | `answer_tag` | closed `<answer>…</answer>` blocks, latest first; then a trailing unclosed `<answer>` |
| 2 | `boxed` | the last `\boxed{…}`, nested braces allowed |
| 3 | `json` | the last `"answer": "…"` (single quotes and unquoted values also accepted) |
| 4 | `phrase` | the last "final answer: X" / "the answer is X" / "answer: X"; X may be on the next line after markdown, e.g. `**Answer:**\nB` |
| 5 | `last_number` | gsm8k/aime25: the last number in the output |
| 5 | `bare_letter` | multiple choice: the last line is just a label (`B`, `**B**`, `"B"`, `Option B.`); code fences are ignored |
| 5 | `choice_phrase` | multiple choice: the last two lines name the answer (see Fix 5) |
| 5 | `last_line` | math500 and other benchmarks: the last non-empty line |
| – | `no_answer` | nothing matched: answer `""`, scored wrong |

Whatever rules 1–4 find is then narrowed down. Inside an `<answer>` block, a `\boxed{}`, JSON or "answer is B" is used. Multiple-choice results become just the letter, and gsm8k/aime25 results become a number.

Details that came from checking real outputs:
- **Closed blocks come first.** Looping distilled models often end with a cut-off `<answer>\n\boxed{1`, and reading that first gave junk.
- **Prose blocks are skipped for numbers.** A multi-line `<answer>` block with no box or phrase is working, not an answer, so the parser moves on to the next rule instead of guessing its last number. A short one-line block like "18 dollars" still counts.
- **The latest block that has an answer wins.** If a model writes `<answer>C</answer>` and then argues for D, the parser keeps C, because the tag is the model's declared answer.

---

## Fix 5: strict multiple-choice letters

### The problem
Without a JSON answer, the old `normalize_choice` took the **first single letter a–e anywhere** in the text, so the English word "a" became answer **A**.

### The fix
A letter only counts when it's clearly given as the answer. Letters must be uppercase, except a lone `b` on its own, and digit labels `1`–`5` are accepted for ARC.

- **Named by "answer":** `the answer is B`, `**Answer:** (C)`, `**Answer**: C`
- **Named by a qualified option word:** `the correct choice is **C**`, `the most plausible ending is D`, `the answer is option B`, `Choose C`
- **"X is the correct/best/most plausible…":** `**C** is the most plausible conclusion`
- **A label on its own:** `B`, `(B)`, `B.`, `B: 4`, `Option B.`

These are **not** read as answers: "a person walks into a room", "Choice C covers the setting, but…", "D is incorrect".

---

## Fix 6: math uses the last number

For gsm8k/aime25, when no tag, box, JSON or phrase gives an answer, the **last** number in the output is used instead of the first. "16 - 3 - 4 = 9, 9 x 2 = 18" now gives `18`, not `16`. Inside a box, phrase or short block, "9 * 2 = 18 dollars" also gives `18`.

math500 is excluded because its answers are LaTeX (`\frac{14}{3}`), where the last number would be `3`.

---

## Fix 7: parsing rule recorded

Every candidate in the output JSON has a `parse_rule` field (one of the rules in the Fix 4 table). Each benchmark summary has a `parse_rules` count, which is also printed after the score:

```
Score: 0.7400 (148/200)
Parse rules: {'boxed': 181, 'answer_tag': 9, 'unfinished_thinking': 8, 'no_answer': 2}
```

- **High `unfinished_thinking`:** the token budget is too small.
- **High `no_answer`:** the prompt format doesn't match how the model answers.

---

## Fix 8: token budget for thinking runs

`final-run-evals/qwen_2B-think-new.yaml` now uses `max_new_tokens: 16384` for every benchmark. It previously used 4096 for gsm8k/mmlu and 1024 for arc-c/hellaswag. Real reasoning often runs longer than those limits, and with Fix 3 a cut-off answer scores as no answer. Adjust this based on the `unfinished_thinking` count. The no-think config is unchanged.

---

## How this was verified
- **Parser:** 59 hand-written outputs run through `extract_answer` and the scorers, covering every rule, the old failure modes, thinking and cut-off reasoning, looping and cut-off tags, nested boxes, markdown answers, comma numbers and ARC digit labels. All 59 pass.
- **Generation:** the repo's `generate_candidates_batch` was run with the real Qwen3.5-2B tokenizer and a stand-in model returning known replies.
  - Short and long prompts in one batch both return exactly the model's reply.
  - Thinking prompts end in `<think>\n` and non-thinking prompts in the empty think block.
  - Draft answers in the reasoning are ignored.
  - Unclosed reasoning is marked `unfinished_thinking`.
- **Full pipeline:** `run_evaluation` was run with the stand-in model in both modes. `parse_rule` and `parse_rules` appear in the saved JSON.
- **Real outputs:** all 31,000 stored outputs in `eval_jsons/` were re-parsed. Where the new parser marks wrong an answer the old parser marked right (with the leak removed), the causes are:
  - multiple choice: in 101 of 102 `answer_tag` cases, the model's `<answer>` tag holds a single wrong letter, so the old parser was right by luck
  - cut-off or looping outputs with no real answer
  - on gsm8k, 19 of 9,000 rows

---

## Not changed
- **Prompts** (`prompts/templates.py`). The prompt-format comparison is a separate step.
- **Sampling settings.** Temperature is still 1.0 with one sample per question, so scores move by a few points between runs.
- **`presence_penalty`.** It is set in the YAMLs, but Hugging Face `generate` doesn't support it, so it has never been applied.
- **`QwenChatFormatter.extract_response` and `create_generation_kwargs`.** Both are unused by the harness and left in place.
