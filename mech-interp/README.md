# Mechanistic Interpretability Experiments for ReasonDistill Students

This folder defines a reproducible experiment contract for testing whether
ReasonDistill causes teacher-selected reasoning features to become represented
and causally used by the student model.

The supported model types are:

```text
reasonDistill
base
logitKD
```

The supported interventions are:

```text
feature_recovery
activation_patching
feature_ablation
feature_steering
```

An implementation agent must receive exactly one `model_type` and exactly one
`intervention` per run. It must not silently substitute a checkpoint, layer,
SAE, or intervention type. If a required path is missing, fail with a clear
error and print the expected input.

---

## 1. Scientific question and interpretation standard

The central hypothesis is:

> ReasonDistill embeds teacher-selected reasoning features into the student’s
> internal representation, and the student uses those features causally during
> reasoning.

Evidence should be reported at four levels:

1. **Representational:** the student contains features that predict the
   teacher’s selected SAE features.
2. **Selective:** those features activate more strongly on reasoning-relevant
   inputs than on matched controls.
3. **Causal:** suppressing or patching those features changes reasoning
   behavior.
4. **Specific:** the effect is stronger in ReasonDistill than in the base and
   logit-KD students, while unrelated capabilities are less affected.

Feature correlation alone is not causal evidence. Every intervention must have
random-feature, random-direction, or matched-norm controls as appropriate.

---

## 2. Repository and model conventions

### 2.1 Student checkpoints

The agent should accept explicit paths or Hugging Face references. Suggested
defaults for this repository are:

```yaml
reasonDistill:
  model: hf://Khaquan/qwen-khaquanS-distillations/qwen_reasondistill_final_continue/epoch_1
  aligner: hf://Khaquan/qwen-khaquanS-distillations/qwen_reasondistill_final_continue/epoch_1/aligner.pt

base:
  model: Qwen/Qwen3.5-2B
  aligner: null

logitKD:
  model: ./checkpoints/qwen_logit_kd_continue/epoch_1
  aligner: null
```

The exact model path must be recorded in the run configuration. The
`aligner.pt` file is an auxiliary ReasonDistill measurement head; it is not a
student transformer checkpoint and must not be treated as an SAE.

### 2.2 Student SAE

Train and select one SAE for each student checkpoint type:

```text
student hidden width: 2048
student SAE expansion: 16x
student SAE latent width: 32768
student SAE weights: untied encoder and decoder
student SAE input layer: layer 12 configuration, subject to the index note below
```

The SAE checkpoint must contain at least:

```text
sae.pt
config.json
trainer_state.pt     # optional for analysis, required for SAE resume
```

The teacher SAE and teacher reason-score file are also required for feature
matching:

```yaml
teacher_sae: hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/sae.pt
teacher_reason_scores: hf://Khaquan/qwen-khaquanS-distillations/qwen-3.5-4B-L16_16x-SAE/reasonscore.pt
teacher_layer: 16
teacher_reasoning_feature_count: 512
```

The top-k teacher feature indices come from `reasonscore.pt`, using the first
`teacher_reasoning_feature_count` entries in `sorted_indices`.

### 2.3 Hidden-state indexing warning

The repository has two layer-index conventions that must be made explicit in
every experiment:

- `training/sae_trainer.py` converts the configured SAE layer to
  `hidden_states[layer + 1]` because hidden-state index 0 is the embedding.
- `training/reason_distill_trainer.py` directly uses
  `hidden_states[student_align_layer]`.

Therefore, an implementation must record both:

```yaml
student_configured_layer: 12
student_hidden_state_index: 13       # used by the current SAE trainer
reason_distill_aligner_state_index: 12 # used by the current training loss
```

Before running experiments, verify which hidden-state index was used to train
the specific student SAE. Do not compare SAE features to the ReasoningFeatureHead
without recording this possible one-index offset.

### 2.4 Student SAE versus ReasoningFeatureHead

The `ReasoningFeatureHead` maps student activations to 512 teacher-feature
predictions. It is not an SAE and is not used during ordinary generation.

Use it as an auxiliary diagnostic for the ReasonDistill student only. Causal
patching, ablation, and steering must modify the actual student hidden state or
student SAE latent representation, not only the output of `ReasoningFeatureHead`.

If the base or logit-KD model has no aligner, feature matching must use the
student SAE latents directly.

---

## 3. Required experiment interface

An implementation should expose a single entry point with an interface similar
to:

```bash
python mech-interp/run.py \
  --model-type reasonDistill \
  --intervention feature_recovery \
  --config mech-interp/configs/<run>.yaml
```

Allowed values:

```text
--model-type {reasonDistill,base,logitKD}
--intervention {feature_recovery,activation_patching,feature_ablation,feature_steering}
```

The implementation may split this into separate modules, but the selected
model and intervention must be visible in every output artifact.

### 3.1 Common configuration

Every experiment config should contain:

```yaml
run:
  name: rd_feature_recovery_v1
  seed: 42
  output_dir: ./mech-interp/results
  overwrite: false
  resume: true

model:
  type: reasonDistill
  checkpoint: <student checkpoint>
  device: cuda
  dtype: bfloat16
  trust_remote_code: true

sae:
  student_checkpoint: <student SAE directory or sae.pt>
  teacher_checkpoint: <teacher SAE directory or sae.pt>
  reason_score_path: <teacher reasonscore.pt>
  teacher_feature_count: 512
  student_layer: 12
  student_hidden_state_index: 13
  teacher_layer: 16
  normalization: match_training_preprocessing

data:
  dataset: amdeepseek
  split: validation
  cache_dir: ./cache
  max_samples: 1000
  max_length: 2048
  batch_size: 1
  shuffle: false

evaluation:
  teacher_forcing: true
  generation: false
  save_activation_shards: true
  shard_size: 100
```

Primary metrics should use teacher forcing and fixed token positions. Free
generation may be added as a secondary result, but it must not replace the
teacher-forced measurement because generation trajectories diverge across
models.

### 3.2 Common output layout

Use one directory per model and intervention:

```text
mech-interp/results/
  <model_type>/
    <intervention>/<run_name>/
      run_config.yaml
      manifest.json
      summary.json
      metrics.jsonl
      per_example.jsonl
      feature_matches.csv       # when feature matching is used
      top_activations.jsonl      # when feature inspection is used
      activation_shards/         # optional, resumable cache
      checkpoints/               # optional intervention state
```

`manifest.json` must record:

```json
{
  "model_type": "reasonDistill",
  "intervention": "feature_recovery",
  "model_checkpoint": "...",
  "student_sae_checkpoint": "...",
  "teacher_sae_checkpoint": "...",
  "teacher_reason_score_path": "...",
  "student_hidden_state_index": 13,
  "teacher_hidden_state_index": 17,
  "seed": 42,
  "dataset_split": "validation",
  "git_commit": "...",
  "status": "running"
}
```

Update `status` to `completed`, `failed`, or `interrupted`. The run must be
resumable without duplicating completed examples.

---

## 4. Shared preprocessing and feature matching

All four experiments use the following shared preparation steps.

### Step A: Load and validate models

1. Load the selected student checkpoint in evaluation mode.
2. Load the student SAE and freeze it.
3. Load the teacher model, teacher SAE, and reason-score file.
4. Load the tokenizer associated with the selected student checkpoint.
5. Verify that all models use the expected tokenizer and vocabulary.
6. Verify the student hidden size matches the student SAE input dimension.
7. Verify the teacher SAE input dimension matches the teacher hidden size.
8. Verify that decoder and encoder weights are untied if that is recorded in
   the SAE config.

### Step B: Build teacher targets

For each tokenized example:

1. Run the teacher with `output_hidden_states=True`.
2. Select the configured teacher hidden-state index.
3. Apply the same normalization used by `ReasoningAlignmentLoss`:
   - convert to float32 for preprocessing;
   - scale each token vector to `target_norm` when applicable;
   - encode with the frozen teacher SAE;
   - apply decoder-column norm scaling when reproducing ReasonDistill targets.
4. Select the first `teacher_feature_count` indices from `sorted_indices`.
5. Save the selected teacher feature matrix and token metadata in a shard.

Teacher targets must be computed once and reused across the three student
models wherever possible.

### Step C: Encode student features

1. Run the selected student with the same input tokens.
2. Capture the hidden state used by the student SAE.
3. Encode it with the frozen student SAE.
4. Store valid-token masks and original `(example_id, token_position)` pairs.

Never compare flattened arrays unless the example ID and token position are
also aligned.

### Step D: Match student SAE features to teacher reasoning features

Student SAE indices and teacher SAE indices are unrelated. Never compare raw
feature index numbers across models.

For each student SAE feature `i` and teacher reasoning feature `j`, calculate on
held-out valid tokens:

```text
value correlation:       corr(z_student_i, t_teacher_j)
presence correlation:    corr(1[z_student_i > threshold],
                               1[t_teacher_j > threshold])
rank correlation:        Spearman(student_i, teacher_j)
top-activation overlap:  overlap(top-N student tokens, top-N teacher tokens)
```

Use a one-to-one assignment such as Hungarian matching for the primary feature
map. Also report many-to-one matches because a student SAE may split one
teacher feature into several features. Save:

```text
teacher_feature_index
student_feature_index
value_correlation
presence_f1
spearman_correlation
top_activation_overlap
assignment_method
```

For ReasonDistill, additionally compare the student SAE feature activations to
the saved `ReasoningFeatureHead` outputs. This measures whether the SAE found
student features that correspond to the supervised readout.

---

## 5. Experiment A: Feature recovery / representational alignment

### Goal

Test whether a student SAE contains features that recover the teacher’s
selected reasoning features, and whether recovery is stronger in ReasonDistill
than in base or logit-KD.

### Required inputs

```text
student checkpoint
student SAE checkpoint
teacher checkpoint
teacher SAE checkpoint
teacher reasonscore.pt
held-out tokenized examples
optional ReasonDistill aligner.pt
```

### Procedure

1. Run shared preprocessing and feature matching.
2. Compute per-feature and aggregate matching metrics.
3. Stratify by:
   - prompt type;
   - token position;
   - reasoning versus non-reasoning tokens when tags are available;
   - teacher feature activation strength.
4. For the top matched features, save the highest-activating text spans.
5. For ReasonDistill, report three comparisons:
   - teacher target versus aligner prediction;
   - teacher target versus student SAE feature;
   - aligner prediction versus student SAE feature.
6. Repeat matching with shuffled token positions and random teacher features.

### Metrics

Required aggregate metrics:

```text
mean_absolute_correlation
median_absolute_correlation
mean_presence_f1
mean_spearman
mean_top_activation_overlap
matched_feature_count_above_threshold
student_feature_sparsity
```

Also report recall at correlation thresholds, for example:

```text
fraction of teacher features with |r| >= 0.1
fraction with |r| >= 0.3
fraction with |r| >= 0.5
```

### Controls

- Random teacher feature indices matched in count.
- Shuffled token positions.
- Random student SAE feature sets.
- Same student SAE and data for all controls.
- Same number of tokens and identical masking.

### Expected interpretation

Higher recovery in ReasonDistill supports representational embedding, but does
not prove causal use. A result is weak if it disappears after token alignment,
is no stronger than random features, or is explained by generic math-token
frequency.

### Saved results

```text
feature_matches.csv
top_activations.jsonl
metrics.jsonl
summary.json
```

---

## 6. Experiment B: Activation patching

### Goal

Test whether reasoning-relevant information can be transferred between clean
and corrupted student forward passes by patching the student residual stream or
student SAE feature subspace.

### Dataset design

Use paired examples with a known behavioral contrast:

```text
clean:     problem with correct reasoning trajectory or answer
corrupted: minimally edited problem with an incorrect trajectory or answer
```

Prefer minimal pairs that preserve length and surface form. Record the expected
correct answer token or answer span for each pair.

### Patch modes

Implement at least two modes:

#### Full hidden-state patch

At a selected `(layer, token_position)`, replace the corrupted student hidden
state with the clean hidden state. This is a sanity check for ordinary causal
tracing.

#### Student SAE latent patch

Let `h_c` be the corrupted hidden state, `z_c` its student SAE latent code, and
`z_clean` the clean latent code. For a selected feature set `S`, patch using:

```text
h_patched = h_c + D_S @ (z_clean_S - z_c_S)
```

where `D_S` contains the corresponding student SAE decoder directions. Preserve
all unselected latent coordinates. If decoder normalization is used, record
whether the patch uses raw or normalized decoder columns.

### Procedure

1. Run clean and corrupted examples and cache the selected hidden states.
2. Compute clean and corrupted student SAE latents.
3. Select feature sets:
   - matched teacher-reasoning features;
   - all matched features;
   - random matched-size features;
   - non-reasoning features.
4. Patch one layer and token position at a time.
5. Sweep reasoning positions and a small layer window around the SAE layer.
6. Run the patched forward pass with teacher forcing.
7. Measure answer-token and reasoning-token effects.

### Metrics

For a metric `m`, define normalized patch recovery as:

```text
recovery = (m_patched - m_corrupted) /
           (m_clean - m_corrupted + epsilon)
```

Required metrics:

```text
clean_correct_answer_logit_margin
corrupted_correct_answer_logit_margin
patched_correct_answer_logit_margin
clean_nll
corrupted_nll
patched_nll
normalized_recovery
```

Also record whether the patch changes the model’s final answer under greedy
decoding, but use teacher-forced metrics as primary evidence.

### Controls

- Full hidden-state patch.
- Random feature patch with matched norm.
- Patch at a random token position.
- Sign-reversed feature patch.
- Patching the same feature set in base and logit-KD.
- Corruption that changes formatting but not the answer.

### Expected interpretation

High recovery from a small matched feature set, combined with low recovery from
random features, supports a causal role. Denoising alone is insufficient when
the model has redundant paths; also run noising patches from clean to corrupted
when possible.

### Saved results

```text
per_example.jsonl
patch_effects.csv
metrics.jsonl
summary.json
```

Each row must include model type, pair ID, layer, token position, patch mode,
feature-set ID, feature count, activation norm, logit margin, NLL, and recovery.

---

## 7. Experiment C: Top-k feature ablation

### Goal

Test whether suppressing student SAE features matched to teacher reasoning
features selectively harms reasoning behavior.

### Feature sets

Run the following ablation sizes:

```text
1, 8, 32, 128, 512
```

For each size, construct:

```text
matched_reasoning_set
random_matched_size_set
non_reasoning_set
highest_activation_set
```

The primary set is the top-k teacher reasoning features mapped to student SAE
features. If fewer than 512 reliable matches exist, report the available count
instead of silently filling the set with arbitrary features.

### Ablation operator

For student hidden state `h`, SAE latent `z`, decoder matrix `D`, and selected
features `S`:

```text
z_ablated[S] = 0
h_ablated = h + D @ (z_ablated - z)
```

Also implement mean ablation and resampling ablation when feasible. Zero
ablation is the primary result, but resampling is useful because it tests
whether the model depends on the particular feature value rather than merely
the feature’s scale.

### Procedure

1. Run unmodified student inference and save baseline metrics.
2. Capture the student SAE layer activation.
3. Encode the activation and ablate the selected latent coordinates.
4. Continue the forward pass and compute logits.
5. Repeat for every ablation set and size.
6. Evaluate reasoning tasks and unrelated control tasks.
7. Repeat across reasoning-token positions and answer-token positions.

### Metrics

Required:

```text
delta_correct_answer_logit_margin
delta_reasoning_token_nll
delta_final_answer_nll
delta_exact_match
relative_reasoning_degradation
relative_control_task_degradation
feature_activation_reduction
```

Report selectivity as:

```text
selectivity = reasoning_degradation - control_task_degradation
```

### Controls

- Random student SAE feature sets with identical cardinality.
- Matched-norm random residual directions.
- Non-reasoning teacher SAE features.
- Ablation at unrelated positions.
- Base and logit-KD students.
- Mean/resampling ablation where possible.

### Expected interpretation

A strong result is a monotonic or dose-dependent degradation of reasoning in
ReasonDistill, with substantially weaker degradation for random features and
control tasks. A global collapse under every ablation indicates an overly large
or poorly localized intervention, not a specific reasoning mechanism.

### Saved results

```text
ablation_effects.csv
per_example.jsonl
metrics.jsonl
summary.json
```

---

## 8. Experiment D: Feature steering

### Goal

Test whether increasing or decreasing student SAE features matched to teacher
reasoning features predictably changes reasoning behavior.

### Steering operators

For selected student decoder directions `D_S`, use either:

```text
h_steered = h + lambda * D_S @ scale
```

or latent-space steering:

```text
z_steered[S] = z[S] + lambda * scale[S]
h_steered = h + D @ (z_steered - z)
```

Prefer latent-space steering because it explicitly controls the selected SAE
features. Use `scale[S]` based on a held-out activation statistic, such as the
per-feature 95th percentile or standard deviation, and record it in the run
manifest.

### Coefficient sweep

Use a symmetric sweep around zero:

```text
lambda ∈ {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}
```

If this causes saturation or instability, calibrate to feature standard
deviations and use a smaller sweep. Never report only the most favorable
coefficient.

### Procedure

1. Establish the unsteered baseline.
2. Compute matched student feature sets.
3. Inject positive and negative steering at the SAE layer.
4. Sweep coefficient, token position, and optionally a small layer window.
5. Evaluate reasoning and control tasks.
6. Test reversibility: positive steering followed by equal negative steering
   should approximately return to baseline.
7. Save the exact activation norm and decoder scaling used for every run.

### Metrics

Required:

```text
correct_answer_logit_margin
reasoning_token_nll
final_answer_nll
exact_match
feature_activation_delta
control_task_delta
activation_norm_delta
```

Derived plots/results:

```text
performance versus lambda
feature activation versus lambda
reasoning selectivity versus lambda
positive/negative symmetry
```

### Controls

- Random matched-norm directions.
- Random SAE feature sets of identical size.
- Non-reasoning feature sets.
- Sign-reversed steering.
- Steering at random token positions.
- Same coefficient and normalization across model types.

### Expected interpretation

Evidence for a reasoning representation requires a predictable, selective
dose-response curve. A direction that changes every task equally, or only
changes output style, should not be labeled a reasoning feature.

### Saved results

```text
steering_sweep.csv
per_example.jsonl
metrics.jsonl
summary.json
```

---

## 9. Evaluation metrics and statistical reporting

Every experiment must report both aggregate and per-example results.

At minimum:

```text
mean
median
standard deviation
standard error or bootstrap confidence interval
number of examples
number of valid tokens
number of failed/skipped examples
```

Use paired bootstrap resampling over the same examples for comparisons between
ReasonDistill, base, and logit-KD. Do not compare models on different examples
or different token masks.

For each primary effect, report:

```text
effect_size
95% confidence_interval
control_effect_size
model_type_difference
```

The summary must explicitly label results as:

```text
representational evidence
causal evidence
selectivity evidence
negative result
inconclusive
```

Do not use the word “proves.” The strongest defensible conclusion is that the
intervention provides evidence for or against a specified causal hypothesis.

---

## 10. Checkpointing and resumability

The implementation must checkpoint analysis progress, not model weights.

Recommended checkpoint contents:

```text
checkpoints/
  state.json
  completed_example_ids.json
  feature_match_state.pt
  activation_shards/
```

`state.json` should include:

```json
{
  "next_example_index": 400,
  "completed_shards": [0, 1, 2, 3],
  "rng_state_saved": true,
  "status": "running"
}
```

On resume:

1. Load and validate `run_config.yaml`.
2. Refuse to resume if model, SAE, tokenizer, layer, seed, or dataset split
   differs unless `--force-resume` is explicitly provided.
3. Skip completed example IDs.
4. Append to JSONL/CSV outputs without duplicating rows.
5. Mark the manifest completed only after all requested metrics are written.

---

## 11. Implementation checklist for a coding agent

Before running:

- [ ] Confirm exactly one model type.
- [ ] Confirm exactly one intervention.
- [ ] Resolve and validate the student checkpoint.
- [ ] Resolve and validate the student SAE.
- [ ] Resolve teacher SAE and `reasonscore.pt`.
- [ ] Confirm hidden-state indices and layer convention.
- [ ] Confirm tokenizer and vocabulary.
- [ ] Confirm dataset split and max sample count.
- [ ] Create the output directory and immutable run manifest.
- [ ] Record the current Git commit.

During running:

- [ ] Use deterministic ordering and fixed seed.
- [ ] Save activation shards incrementally.
- [ ] Log skipped/failed examples.
- [ ] Log activation norms and feature counts for every intervention.
- [ ] Save a progress checkpoint after every shard.
- [ ] Avoid loading all activations into GPU memory at once.

After running:

- [ ] Write `summary.json`.
- [ ] Write aggregate confidence intervals.
- [ ] Verify row counts and unique example IDs.
- [ ] Verify control experiments completed.
- [ ] Mark manifest status as `completed`.
- [ ] Print output paths and the primary interpretation.

---

## 12. Minimum acceptance criteria

An experiment is considered complete only if:

1. The selected model and SAE checkpoints are recorded.
2. The exact hidden-state index is recorded.
3. The primary intervention and at least one matched control ran.
4. Results contain per-example and aggregate metrics.
5. The run is reproducible from `run_config.yaml`.
6. No conclusion relies exclusively on a probe or correlation when a causal
   intervention was requested.
7. ReasonDistill results are compared against at least one of base or logit-KD
   using the same evaluation examples.

The four experiments are complementary:

```text
feature recovery  → is the feature represented?
patching          → can the information be transferred causally?
ablation          → is the feature necessary?
steering          → is the feature sufficient or controllable?
```

