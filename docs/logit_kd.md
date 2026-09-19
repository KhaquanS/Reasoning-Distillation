# Qwen logit KD baseline

This baseline uses the existing `scripts/train.py`, `LogitKDTrainer`, model loaders,
dataset formatting, CSV logging, and checkpoint format. The teacher is
`Qwen/Qwen3.5-4B` (frozen, 8-bit); the student is `Qwen/Qwen3.5-2B` (full fine-tuning).
No SAE, reasoning scores, or alignment head are needed.

Both stages match the corresponding final ReasonDistill configs. The alignment
term is removed; the KD and CE weights are preserved without renormalizing them.

| Setting | Initial stage | Continuation |
| --- | --- | --- |
| Reference | `qwen_reasondistill_final.yaml` | `qwen_reasondistill_final_continue.yaml` |
| AM-DeepSeek `am_0.9M` examples | First 40,000 | Next 12,032 (skip 40,000) |
| Student initialization | Base Qwen student | Initial KD stage's `epoch_1` |
| Epochs / max length | 1 / 2,048 | 1 / 2,048 |
| Batch size / accumulation | 1 / 128 | 1 / 128 |
| Optimizer steps | 313 | 94 |
| Peak / minimum LR | 2e-5 / 4e-6 | 3e-5 / 1.5e-5 |
| Scheduler | Cosine restarts, interval 39 | Cosine restarts, interval 31 |
| Temperature | 4.0 | 4.0 |
| KD / CE weights | 0.30 / 0.90 | 0.30 / 0.75 |

The objective is `alpha_kd * T² * KL(teacher/T || student/T) + beta_ce * CE`.
KL is averaged over attention-mask-valid positions. CE uses the repo's shared
next-token loss and ignores the tokenizer's padding ID. Both losses use the same
concatenated instruction/response sequences as ReasonDistill. The continuation
loads student weights and starts a new optimizer and schedule, matching the
existing ReasonDistill continuation behavior; it is not an exact interrupted-run
resume.

## Run on Vast.ai

In a CUDA instance with this repository available, start from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
bash scripts/train_logit_kd.sh
```

The launcher runs both stages sequentially and stops if either fails. It checks
for CUDA before starting. Use `PYTHON=/path/to/python` to select an interpreter.
The first run downloads the teacher, student, tokenizer, and dataset to `./cache`.
Choose a GPU with enough memory for the 8-bit teacher, full student training,
optimizer states, and 2,048-token activations; the full Qwen run has not been
memory-profiled locally.

Run individual stages if needed:

```bash
bash scripts/train_logit_kd.sh initial
bash scripts/train_logit_kd.sh continue
```

The equivalent direct training commands are:

```bash
python3 scripts/train.py --config configs/qwen_logit_kd.yaml
python3 scripts/train.py --config configs/qwen_logit_kd_continue.yaml
```

Training settings remain in YAML, following the existing training interface.
Continuation expects `./checkpoints/qwen_logit_kd/epoch_1`; a checkpoint-loading
failure stops the KD run instead of silently restarting from base weights.

Outputs:

- Initial student: `checkpoints/qwen_logit_kd/epoch_1`
- Final student: `checkpoints/qwen_logit_kd_continue/epoch_1`
- Loss CSVs: `logs/qwen_logit_kd/training_loss.csv` and
  `logs/qwen_logit_kd_continue/training_loss.csv`
- Intermediate checkpoints: `step_150` and `step_300` in the initial stage;
  `step_45` and `step_90` in the continuation.

Each checkpoint includes the student and tokenizer in Hugging Face format plus
`trainer_state.pt`. Use the final checkpoint as a model entry's `checkpoint` in a
copy of the existing evaluation YAML (remove its Hub `subfolder`, if present).
Re-running a stage uses the same output
paths, so change its YAML checkpoint/log directories to retain a previous run.

## Local validation

```bash
python3 -m unittest discover -s tests -p 'test_logit_kd.py' -v
```

Tests use tiny randomly initialized models on CPU, require the training Python
dependencies, and download no model weights or datasets. They check the loss,
padding, gradient flow, config parity, two-stage training and checkpoint output,
and failure on a missing continuation checkpoint.
