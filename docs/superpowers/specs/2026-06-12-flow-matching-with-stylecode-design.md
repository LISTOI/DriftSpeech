# Flow-Matching with StyleCode Design

## Goal

Create `exp/flow-matching-with-stylecode` as a stylecode-conditioned version of the current no-stylecode flow-mel baseline.

The branch should test whether adding phoneme-level stylecode conditioning improves flow-matching mel generation quality under the same GT-duration evaluation protocol used by the no-stylecode baseline.

## Baseline being extended

Current branch baseline:

```text
text/token_ids
-> FastSpeech2 encoder
-> duration expansion
-> ConvNeXt V2 1D frame smoother
-> flow-matching mel generator
-> generated mel
```

The branch keeps:

- text encoder;
- duration predictor;
- length regulator;
- ConvNeXt V2 frame condition smoother;
- CFG-capable flow-matching mel generator;
- GT-duration flow-mel evaluation sweep.

## Stylecode conditioning architecture

Training path:

```text
token_ids
-> text encoder
-> text_hidden [B, N, H]

mel_targets + duration_targets
-> PhonemeStyleExtractor
-> stylecode [B, N, C]
-> decode_stylecode
-> style_hidden [B, N, H]

text_hidden + style_hidden
-> length regulator using duration_targets
-> frame_hidden [B, T, H]
-> ConvNeXt V2 smoother
-> frame_condition [B, T, H]

Gaussian noise x0 + GT mel x1 + time t
-> flow interpolation x_t
-> MelFlowGenerator(x_t, t, frame_condition)
-> v_pred
```

The main fusion rule is intentionally simple:

```text
phoneme_condition = text_hidden + style_hidden
```

This matches the earlier style extractor integration and keeps the experiment focused on whether stylecode helps the flow-mel generator.

## Inference modes

### GT/reference stylecode mode

Primary first-version mode.

Input:

```text
text + duration + reference mel
```

Flow:

```text
reference mel + duration
-> style extractor
-> stylecode
-> style_hidden
-> text_hidden + style_hidden
-> flow sampler
```

This is the intended evaluation mode for comparing against the no-stylecode baseline.

### Zero-style fallback

If no mel target/reference mel is available:

```text
style_hidden = 0
```

The model falls back to no-stylecode conditioning.

This is useful for debugging but is not the main quality target for this branch.

### Predicted stylecode

Not implemented in this branch. Future work can connect the standalone `StyleCodePredictor` to provide predicted stylecode for text-only synthesis.

## Losses

Total loss:

```text
total_loss =
  flow_mel_weight * flow_mel_loss
+ duration_weight * duration_loss
+ adversarial_weight * phoneme_adv_loss
+ optional_vq_loss
```

First version default:

```text
VQ disabled
adversarial enabled
```

Flow loss:

```text
flow_mel_loss = masked_mse(v_pred, v_target)
v_target = x1 - x0
```

Duration loss:

```text
duration_loss = mse(log_duration_prediction, log1p(duration_target))
```

Adversarial loss:

```text
phoneme_adv_loss = cross_entropy(style_adv_logits, token_ids)
```

Only valid non-padding, positive-duration phoneme positions contribute.

## Recommended first training parameters

Use continuous stylecode with adversarial training and no VQ:

```yaml
phoneme_style:
  frame_hidden: 256
  conv_layers: 2
  conv_kernel_size: 5
  dropout: 0.1
  bottleneck_dim: 32
  strict_duration_check: true
  adversarial:
    enabled: true
    hidden: 64
    dropout: 0.1
    weight: 0.02
    grl_lambda: 0.5
  vq:
    enabled: false
    codebook_size: 128
    commitment_weight: 0.25
    codebook_weight: 1.0
```

Reasoning:

- prior stylecode adversarial-from-scratch run reached KNN around 0.14 at 700k with good synthesis;
- this branch's first goal is quality improvement over no-stylecode flow baseline, not maximum leakage suppression;
- `weight=0.02`, `grl_lambda=0.5` gives moderate adversarial pressure while reducing the risk of over-erasing style information.

## Evaluation protocol

Primary evaluation uses GT duration and GT/reference stylecode:

```text
validation text + GT duration + GT mel for style extraction
-> flow sampler
-> generated mel
-> compare generated mel against GT mel
```

This should be compared against no-stylecode baseline under the same:

- validation sample subset;
- checkpoint list;
- guidance scale list;
- sample steps list;
- vocoder;
- metric table.

Automatic metrics:

```text
mel_l1
mel_mse
mel_rmse
mel_mean_abs_diff
mel_std_abs_diff
score = mel_l1_mean + 0.1 * mel_std_abs_diff_mean
```

Final quality selection should still include listening to generated samples.

## Output paths

Recommended LJSpeech experiment paths:

```yaml
path:
  ckpt_path: "./output/ckpt/LJSpeech_flow_stylecode_adv"
  log_path: "./output/log/LJSpeech_flow_stylecode_adv"
  result_path: "./output/result/LJSpeech_flow_stylecode_adv"
```

## Implementation scope

Implement now:

1. restore `PhonemeStyleExtractor` in the flow baseline model;
2. add style-hidden fusion before length regulation;
3. restore adversarial loss support in `FastSpeech2Loss`;
4. keep VQ code path compatible but disabled by config;
5. update training/evaluation log messages;
6. update flow-mel evaluation script to use GT stylecode extraction when mel targets are present;
7. update configs and documentation;
8. run py_compile and an isolated smoke test.

Do not implement now:

- predicted stylecode inference;
- VQ stylecode experiment;
- style transfer reference selection;
- ASR/PESQ/STOI evaluation.
