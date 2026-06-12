# Flow-Matching with StyleCode Change List

Branch:

```text
exp/flow-matching-with-stylecode
```

Compared against previous no-stylecode flow baseline:

```text
exp/no-stylecode-flow-mel-baseline
```

Diff summary:

```text
13 files changed, 469 insertions(+), 27 deletions(-)
```

## Changed files

```text
M  FastSpeech2/config/AISHELL3/model.yaml
M  FastSpeech2/config/LJSpeech/model.yaml
M  FastSpeech2/config/LJSpeech/train.yaml
M  FastSpeech2/config/LJSpeech_paper/model.yaml
M  FastSpeech2/config/LibriTTS/model.yaml
M  FastSpeech2/evaluate.py
M  FastSpeech2/evaluate_flow_mel.py
M  FastSpeech2/model/fastspeech2.py
M  FastSpeech2/model/loss.py
M  FastSpeech2/train.py
M  FastSpeech2/utils/tools.py
A  docs/superpowers/plans/2026-06-12-flow-matching-with-stylecode.md
A  docs/superpowers/specs/2026-06-12-flow-matching-with-stylecode-design.md
```

## Architecture difference

### Previous no-stylecode flow baseline

```text
text/token_ids
-> FastSpeech2 encoder
-> duration expansion
-> ConvNeXt V2 1D frame smoother
-> flow-matching mel generator
-> generated mel
```

### Current stylecode-conditioned flow baseline

```text
text/token_ids
-> FastSpeech2 encoder
-> text_hidden [B,N,H]

GT/reference mel + duration_targets
-> PhonemeStyleExtractor
-> stylecode [B,N,C]
-> decode_stylecode
-> style_hidden [B,N,H]

text_hidden + style_hidden
-> duration expansion
-> ConvNeXt V2 1D frame smoother
-> flow-matching mel generator
-> generated mel
```

The key change is:

```text
flow generator condition = duration_expand(text_hidden + decoded_style_hidden)
```

instead of:

```text
flow generator condition = duration_expand(text_hidden)
```

## File-by-file changes

### `FastSpeech2/model/fastspeech2.py`

Restores stylecode conditioning in the flow-mel model.

Main changes:

- imports and instantiates `PhonemeStyleExtractor`;
- extracts stylecode from `mels + d_targets` when reference/GT mel is available;
- decodes stylecode to `style_hidden [B,N,H]`;
- fuses style with text by:

```text
phoneme_condition = text_hidden + style_hidden
```

- length-regulates `phoneme_condition` instead of plain `text_hidden`;
- keeps zero-style fallback when no mel/reference is available;
- adds `force_sampling` argument so evaluation can pass GT mel for style extraction while still forcing flow sampling;
- returns `style_adv_logits` and `style_info` after `flow_info`.

New output tuple:

```text
(
  mel_predictions,
  v_pred,
  v_target,
  log_duration_predictions,
  src_masks,
  mel_masks,
  src_lens,
  mel_lens,
  flow_info,
  style_adv_logits,
  style_info,
)
```

### `FastSpeech2/model/loss.py`

Restores style-aware loss terms on top of flow matching.

Main changes:

- keeps `flow_mel_loss`;
- keeps `duration_loss`;
- restores `phoneme_adv_loss` from `style_adv_logits`;
- restores optional VQ losses from `style_info`;
- total loss now supports:

```text
total_loss =
  flow_mel_weight * flow_mel_loss
+ duration_weight * duration_loss
+ adv_weight * phoneme_adv_loss
+ vq_loss
```

Returned losses:

```text
(total_loss, flow_mel_loss, duration_loss, phoneme_adv_loss)
```

If VQ info exists, additional metrics are appended:

```text
vq_loss
vq_commitment_loss
vq_codebook_loss
codebook_perplexity
used_codes
```

### `FastSpeech2/train.py`

Updates training logs and restores stylecode export for this branch.

Main changes:

- training logs support:

```text
Total Loss
Flow Mel Loss
Duration Loss
Phoneme Adv Loss
VQ Loss
VQ Commitment Loss
VQ Codebook Loss
Codebook Perplexity
Used Codes
```

- `dump_stylecodes()` now runs when the model has `style_extractor`;
- stylecode JSONL export is available again for leakage analysis.

### `FastSpeech2/evaluate.py`

Updates validation logging to match the extended loss tuple.

Validation logs now support:

```text
Total Loss
Flow Mel Loss
Duration Loss
Phoneme Adv Loss
VQ metrics when present
```

### `FastSpeech2/utils/tools.py`

Updates TensorBoard scalar logging.

New/active scalar tags:

```text
Loss/total_loss
Loss/flow_mel_loss
Loss/duration_loss
Loss/phoneme_adv_loss
Loss/vq_loss
Loss/vq_commitment_loss
Loss/vq_codebook_loss
VQ/codebook_perplexity
VQ/used_code_count
```

The mel/audio sample helpers remain compatible with the flow prediction tuple.

### `FastSpeech2/evaluate_flow_mel.py`

Updates GT-duration evaluation to support GT/reference stylecode.

Main change:

```text
sample_with_gt_duration(...)
```

now passes:

```text
mels=batch[6]
mel_lens=batch[7]
d_targets=batch[9]
force_sampling=True
```

This means:

```text
GT mel is used to extract stylecode,
but generated mel is still produced by flow sampling from noise.
```

So the evaluation protocol becomes:

```text
validation text + GT duration + GT/reference stylecode
-> flow sampler
-> generated mel
-> compare generated mel with GT mel
```

This is the intended fair comparison against the no-stylecode baseline.

## Config changes

### Model config files

Modified:

```text
FastSpeech2/config/AISHELL3/model.yaml
FastSpeech2/config/LJSpeech/model.yaml
FastSpeech2/config/LJSpeech_paper/model.yaml
FastSpeech2/config/LibriTTS/model.yaml
```

Adversarial stylecode branch changed from weak default:

```yaml
weight: 0.01
grl_lambda: 0.1
```

Conceptually to recommended moderate setting:

```yaml
phoneme_style:
  adversarial:
    enabled: true
    hidden: 64
    dropout: 0.1
    weight: 0.02
    grl_lambda: 0.5
  vq:
    enabled: false
```

The intent is:

```text
use moderate adversarial pressure to reduce phoneme leakage
without over-erasing style information needed by the flow generator
```

### LJSpeech train config

Modified:

```text
FastSpeech2/config/LJSpeech/train.yaml
```

Experiment output paths changed to:

```yaml
path:
  ckpt_path: "./output/ckpt/LJSpeech_flow_stylecode_adv"
  log_path: "./output/log/LJSpeech_flow_stylecode_adv"
  result_path: "./output/result/LJSpeech_flow_stylecode_adv"
```

Stylecode logging re-enabled:

```yaml
stylecode_logging:
  enabled: true
```

This allows checkpoint-time stylecode JSONL export for leakage evaluation.

## Added design/plan docs

### `docs/superpowers/specs/2026-06-12-flow-matching-with-stylecode-design.md`

Design document covering:

- stylecode-conditioned flow architecture;
- training and inference modes;
- loss composition;
- recommended adversarial parameters;
- evaluation protocol;
- implementation scope.

### `docs/superpowers/plans/2026-06-12-flow-matching-with-stylecode.md`

Implementation plan covering:

- model forward changes;
- loss restoration;
- log updates;
- evaluation script update;
- config updates;
- static checks and smoke test.

## Recommended training setup

Recommended first run:

```yaml
phoneme_style:
  adversarial:
    enabled: true
    hidden: 64
    dropout: 0.1
    weight: 0.02
    grl_lambda: 0.5
  vq:
    enabled: false
```

Reason:

```text
The first objective is to test whether continuous stylecode improves flow-mel generation quality.
Moderate adversarial pressure is safer than strong pressure because stylecode is now a useful generation condition.
```

If quality is stable but leakage remains high, try:

```yaml
weight: 0.03
grl_lambda: 0.5
```

Do not start with `weight: 0.04` or `0.05` unless audio quality and style diversity remain stable.

## Training command

Short smoke train:

```bash
cd /home/listoi/Speech/FastSpeech2

CUDA_VISIBLE_DEVICES=0 nohup conda run -n torch_gpu python train.py \
  --restore_step 0 \
  -p config/LJSpeech/preprocess.yaml \
  -m config/LJSpeech/model.yaml \
  -t config/LJSpeech/train.yaml \
  > flow_stylecode_adv_smoke.log 2>&1 &
```

Formal train:

```bash
cd /home/listoi/Speech/FastSpeech2

CUDA_VISIBLE_DEVICES=0 nohup conda run -n torch_gpu python train.py \
  --restore_step 0 \
  -p config/LJSpeech/preprocess.yaml \
  -m config/LJSpeech/model.yaml \
  -t config/LJSpeech/train.yaml \
  > flow_stylecode_adv_train.log 2>&1 &
```

## Evaluation command

```bash
cd /home/listoi/Speech/FastSpeech2

CUDA_VISIBLE_DEVICES=0 conda run -n torch_gpu python evaluate_flow_mel.py \
  --restore_steps 100000,200000,300000 \
  --guidance_scales 1.0,1.5,2.0 \
  --sample_steps 16,32,64 \
  --max_samples 50 \
  --save_samples 10 \
  --save_wav \
  --output_dir output/result/LJSpeech_flow_stylecode_adv/eval_sweep \
  -p config/LJSpeech/preprocess.yaml \
  -m config/LJSpeech/model.yaml \
  -t config/LJSpeech/train.yaml
```

## Verification performed

Static compile check:

```bash
conda run -n torch_gpu python -m py_compile \
  FastSpeech2/model/*.py \
  FastSpeech2/train.py \
  FastSpeech2/evaluate.py \
  FastSpeech2/synthesize.py \
  FastSpeech2/utils/*.py \
  FastSpeech2/evaluate_flow_mel.py
```

Result:

```text
passed with exit code 0
```

Isolated smoke test:

```bash
conda run -n torch_gpu python /tmp/stylecode_flow_smoke.py
```

Latest output:

```text
stylecode flow smoke ok (2, 7, 80) [4.3651, 2.3424, 1.9345, 4.409]
```

Verified locally:

- model forward runs with GT mel style extraction;
- flow output shape is `[B, T_mel, 80]`;
- adversarial logits are produced;
- loss tuple includes phoneme adversarial loss;
- backward pass runs;
- `force_sampling=True` path samples from flow while using GT mel as style reference;
- sampled mel values are finite.

Full training has not been run locally and should be run on the remote GPU server.
