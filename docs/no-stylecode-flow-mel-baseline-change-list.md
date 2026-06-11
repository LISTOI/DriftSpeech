# No-StyleCode Flow-Mel Baseline Change List

Branch:

```text
exp/no-stylecode-flow-mel-baseline
```

Base branch:

```text
main
```

Compared with `main`, this branch changes 17 files:

```text
17 files changed, 929 insertions(+), 168 deletions(-)
```

## Added files

### `FastSpeech2/model/mel_flow.py`

New flow-matching mel generation module.

Main contents:

- `SinusoidalTimeEmbedding`
- `SinusoidalPositionEmbedding`
- `AdaLayerNorm`
- `GlobalResponseNorm1D`
- `ConvNeXtV2Block1D`
- `FrameConditionSmoother`
- `MelFlowBlock`
- `MelFlowGenerator`
- `masked_flow_matching_loss`

Main responsibility:

```text
frame-level text condition + noisy mel x_t + time t
-> predict velocity v_pred for flow matching
```

It also implements CFG support:

```text
condition dropout during training
learned null condition
CFG velocity mixing during Euler sampling
```

### `docs/superpowers/specs/2026-06-05-no-stylecode-flow-mel-baseline-design.md`

Design document for the experiment branch.

Covers:

- architecture goal;
- data flow;
- module boundaries;
- flow-matching objective;
- CFG design;
- training and inference behavior;
- config additions;
- verification plan.

### `docs/superpowers/plans/2026-06-05-no-stylecode-flow-mel-baseline.md`

Implementation plan for the experiment branch.

Covers:

- files to create/modify;
- task breakdown;
- static check and smoke test plan.

### `FastSpeech2/evaluate_flow_mel.py`

GT-duration flow-mel evaluation sweep script.

Main capabilities:

- compares multiple checkpoints;
- sweeps CFG guidance scale;
- sweeps Euler sample steps;
- uses validation-set GT duration for all generations;
- computes frame-aligned mel metrics;
- saves `summary.csv`, `summary.json`, and `best.json`;
- optionally saves generated/GT mel PNG and wav samples.

### `docs/superpowers/plans/2026-06-11-flow-mel-evaluation-sweep.md`

Implementation plan for the GT-duration flow-mel evaluation sweep script.

## Modified model files

### `FastSpeech2/model/fastspeech2.py`

Replaces the original decoder/postnet path with the no-stylecode flow-matching mel path.

Main changes:

- removes style extractor usage from `forward()`;
- removes decoder/postnet mel regression path from this branch;
- keeps text encoder, speaker embedding, duration predictor, and length regulator;
- adds a ConvNeXt V2-style `FrameConditionSmoother` after duration expansion;
- adds `MelFlowGenerator` for mel velocity prediction;
- training path uses ground-truth duration targets and mel targets;
- inference path samples mel from Gaussian noise with Euler integration;
- supports train config injection through `set_train_config()`.

New high-level flow:

```text
text -> encoder -> duration expansion -> ConvNeXt V2 frame smoother -> mel flow generator
```

### `FastSpeech2/model/loss.py`

Replaces original mel/postnet/style/VQ loss logic with flow-matching loss.

Main changes:

- computes `flow_mel_loss = masked_mse(v_pred, v_target)`;
- keeps duration prediction loss;
- returns:

```text
(total_loss, flow_mel_loss, duration_loss)
```

New total loss:

```text
total_loss = flow_mel_weight * flow_mel_loss + duration_weight * duration_loss
```

## Modified training/evaluation/runtime files

### `FastSpeech2/train.py`

Updates training logs and disables stylecode export for this branch.

Main changes:

- training log now reports:

```text
Total Loss, Flow Mel Loss, Duration Loss
```

- `dump_stylecodes()` returns immediately when the model has `mel_flow`, because this branch intentionally disables stylecode extraction.

### `FastSpeech2/evaluate.py`

Updates validation logging for the new loss tuple.

Validation log now reports:

```text
Validation Step, Total Loss, Flow Mel Loss, Duration Loss
```

### `FastSpeech2/utils/tools.py`

Updates TensorBoard logging and sample synthesis helpers for the new prediction tuple.

Main changes:

- TensorBoard scalar tags now use:

```text
Loss/total_loss
Loss/flow_mel_loss
Loss/duration_loss
```

- `synth_one_sample()` continues to read generated mel from `predictions[0]`;
- `synth_samples()` now reads mel lengths from the new tuple index `predictions[7]`.

### `FastSpeech2/utils/model.py`

Injects `train_config` into the model when supported:

```text
model.set_train_config(train_config)
```

This lets inference use flow sampling settings such as:

```text
sample_steps
guidance_scale
```

## Modified config files

The same changes were applied across AISHELL3, LJSpeech, LJSpeech_paper, and LibriTTS configs.

### Model config files

Modified:

```text
FastSpeech2/config/AISHELL3/model.yaml
FastSpeech2/config/LJSpeech/model.yaml
FastSpeech2/config/LJSpeech_paper/model.yaml
FastSpeech2/config/LibriTTS/model.yaml
```

Main additions:

```yaml
mel_flow:
  enabled: true
  hidden_dim: 256
  depth: 6
  num_heads: 4
  dropout: 0.1
  conv_smoother_layers: 4
  conv_smoother_kernel_size: 7
  conv_smoother_expansion: 4
  time_embed_dim: 256
  noise_scale: 1.0
  prediction_type: velocity
  cfg:
    enabled: true
    condition_dropout: 0.1
    null_condition: learned
    guidance_scale: 1.5
```

Loss config now includes:

```yaml
loss:
  duration_weight: 1.0
  flow_mel_weight: 1.0
```

### Train config files

Modified:

```text
FastSpeech2/config/AISHELL3/train.yaml
FastSpeech2/config/LJSpeech/train.yaml
FastSpeech2/config/LJSpeech_paper/train.yaml
FastSpeech2/config/LibriTTS/train.yaml
```

Main changes:

```yaml
stylecode_logging:
  enabled: false
```

Added:

```yaml
mel_flow:
  sample_steps: 32
  guidance_scale: 1.5
```

## Behavior difference from `main`

### `main`

```text
text encoder
-> duration predictor
-> style extractor from mel target
-> stylecode decode
-> text_hidden + style_phoneme
-> length regulator
-> decoder/postnet
-> mel regression loss + duration/style/VQ losses
```

### `exp/no-stylecode-flow-mel-baseline`

```text
text encoder
-> duration predictor
-> length regulator using duration targets during training
-> frame-level text condition
-> ConvNeXt V2 1D frame smoother
-> flow-matching mel generator
-> flow velocity loss + duration loss
```

The experiment branch does not use stylecode extraction in the model forward path.

## Flow-mel evaluation mechanism

The branch now includes a GT-duration sweep script:

```bash
conda run -n torch_gpu python evaluate_flow_mel.py \
  --restore_steps 100000,200000,300000 \
  --guidance_scales 1.0,1.5,2.0 \
  --sample_steps 16,32,64 \
  --max_samples 50 \
  --save_samples 10 \
  --save_wav \
  --output_dir output/result/LJSpeech_flow_mel_baseline/eval_sweep \
  -p config/LJSpeech/preprocess.yaml \
  -m config/LJSpeech/model.yaml \
  -t config/LJSpeech/train.yaml
```

Evaluation protocol:

```text
validation text + GT MFA duration
-> flow sampler
-> generated mel
-> compare against GT mel over valid frames
```

Metrics:

```text
mel_l1
mel_mse
mel_rmse
mel_mean_abs_diff
mel_std_abs_diff
```

Automatic ranking score:

```text
score = mel_l1_mean + 0.1 * mel_std_abs_diff_mean
```

The automatic best combination is written to:

```text
best.json
```

The ranking is intended for candidate selection. Final model choice should still be confirmed by listening to the saved generated wav samples.

## Verification performed

Static compile check:

```bash
conda run -n torch_gpu python -m py_compile \
  FastSpeech2/model/*.py \
  FastSpeech2/train.py \
  FastSpeech2/evaluate.py \
  FastSpeech2/synthesize.py \
  FastSpeech2/utils/*.py
```

Result:

```text
passed with exit code 0
```

Isolated flow-mel smoke test:

```bash
conda run -n torch_gpu python /tmp/flow_mel_smoke.py
```

Verified locally:

- forward path runs;
- output mel shape is `[B, T_mel, 80]`;
- flow loss is finite;
- duration loss is finite;
- one backward pass runs;
- Euler inference sampler returns finite mel output;
- CFG path has no shape/runtime error.

Latest smoke output:

```text
flow mel smoke ok (2, 7, 80) [4.5567, 2.5226, 2.0341]
```

Full training has not been run locally. Full training should be run on the remote GPU server.
