# No-StyleCode Flow-Mel Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FastSpeech2 decoder/postnet path with a no-stylecode flow-matching mel generator conditioned on duration-expanded text features.

**Architecture:** The model keeps the FastSpeech2 encoder, speaker embedding, duration predictor, and length regulator. It removes stylecode usage from the forward path, smooths frame-level text conditions with 1D convolutions, and trains a sequence DiT-style mel flow generator with optional classifier-free guidance.

**Tech Stack:** PyTorch, existing FastSpeech2 dataset/train/evaluate/synthesize scripts, YAML config.

---

## File Structure

- Create `FastSpeech2/model/mel_flow.py`: time embedding, AdaLN, frame condition smoother, flow blocks, generator, masked flow matching loss, Euler sampler with CFG.
- Modify `FastSpeech2/model/fastspeech2.py`: remove style extractor from forward path, replace decoder/postnet with flow generator, add inference sampler path.
- Modify `FastSpeech2/model/loss.py`: compute flow mel loss + duration loss.
- Modify `FastSpeech2/train.py`: disable stylecode dumping for this branch and update logging message.
- Modify `FastSpeech2/evaluate.py`: update validation logging message.
- Modify `FastSpeech2/utils/tools.py`: update scalar tags and make sample synthesis work with flow predictions.
- Modify `FastSpeech2/config/*/model.yaml`: add `mel_flow` block and `flow_mel_weight`.
- Modify `FastSpeech2/config/*/train.yaml`: disable stylecode logging and add flow sample steps.

## Tasks

### Task 1: Add mel flow module

**Files:**
- Create: `FastSpeech2/model/mel_flow.py`

- [ ] Implement `SinusoidalTimeEmbedding`, `AdaLayerNorm`, `FrameConditionSmoother`, `MelFlowBlock`, `MelFlowGenerator`, `masked_flow_matching_loss`, and `euler_sample`.
- [ ] Ensure masks use existing convention: `True` means padding.
- [ ] Ensure CFG supports learned null condition, training condition dropout, and inference guidance scale.
- [ ] Run `python -m py_compile FastSpeech2/model/mel_flow.py`.

### Task 2: Replace FastSpeech2 forward path

**Files:**
- Modify: `FastSpeech2/model/fastspeech2.py`

- [ ] Import `FrameConditionSmoother` and `MelFlowGenerator`.
- [ ] Remove style extractor usage from `forward()`.
- [ ] Keep encoder, speaker embedding, duration predictor, and length regulator.
- [ ] Training path returns `(mel_predictions, flow_target, log_duration_predictions, src_masks, mel_masks, src_lens, mel_lens, flow_info)`.
- [ ] Inference path samples mel from Gaussian noise using Euler sampler.
- [ ] Run `python -m py_compile FastSpeech2/model/fastspeech2.py`.

### Task 3: Update loss for flow matching

**Files:**
- Modify: `FastSpeech2/model/loss.py`

- [ ] Compute masked MSE between `v_pred` and `v_target` over valid mel frames.
- [ ] Keep duration loss over valid source positions.
- [ ] Return `(total_loss, flow_mel_loss, duration_loss)`.
- [ ] Run `python -m py_compile FastSpeech2/model/loss.py`.

### Task 4: Update logging and sampling helpers

**Files:**
- Modify: `FastSpeech2/train.py`
- Modify: `FastSpeech2/evaluate.py`
- Modify: `FastSpeech2/utils/tools.py`

- [ ] Update loss messages to `Total Loss`, `Flow Mel Loss`, `Duration Loss`.
- [ ] Disable stylecode dump when `model_config["mel_flow"]["enabled"]` is true.
- [ ] Keep TensorBoard mel/audio sample generation compatible with predictions tuple.
- [ ] Run `python -m py_compile FastSpeech2/train.py FastSpeech2/evaluate.py FastSpeech2/utils/tools.py`.

### Task 5: Update configs

**Files:**
- Modify: `FastSpeech2/config/AISHELL3/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech_paper/model.yaml`
- Modify: `FastSpeech2/config/LibriTTS/model.yaml`
- Modify: `FastSpeech2/config/AISHELL3/train.yaml`
- Modify: `FastSpeech2/config/LJSpeech/train.yaml`
- Modify: `FastSpeech2/config/LJSpeech_paper/train.yaml`
- Modify: `FastSpeech2/config/LibriTTS/train.yaml`

- [ ] Add `mel_flow` config to model YAML files.
- [ ] Add `loss.flow_mel_weight` to model YAML files.
- [ ] Disable `stylecode_logging.enabled` in train YAML files.
- [ ] Add `mel_flow.sample_steps: 32` to train YAML files.

### Task 6: Static checks and smoke test

**Files:**
- Run against modified Python files.

- [ ] Run py_compile for model, train, evaluate, synthesize, utils.
- [ ] Run a minimal import/shape smoke test if dependencies are available.
- [ ] Commit implementation changes if checks pass.
