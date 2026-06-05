# ConvNeXt V2 Condition Smoother Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple 1D convolution condition smoother with a ConvNeXt V2-style 1D smoother for the no-stylecode flow-mel baseline.

**Architecture:** Keep the external `FrameConditionSmoother` interface so `FastSpeech2/model/fastspeech2.py` does not need structural changes. Internally use stacked 1D ConvNeXt V2 blocks: depthwise temporal convolution, LayerNorm, pointwise MLP expansion, GRN, projection, dropout, residual, and padding mask.

**Tech Stack:** PyTorch, existing `FastSpeech2/model/mel_flow.py`, YAML configs.

---

## File Structure

- Modify `FastSpeech2/model/mel_flow.py`: add `GlobalResponseNorm1D` and `ConvNeXtV2Block1D`; replace `FrameConditionSmoother` internals.
- Modify `FastSpeech2/config/*/model.yaml`: set smoother defaults to 4 layers, kernel size 7, expansion 4.
- Modify `docs/no-stylecode-flow-mel-baseline-change-list.md`: document ConvNeXt V2 smoother.

## Tasks

### Task 1: Implement ConvNeXt V2 smoother classes

**Files:**
- Modify: `FastSpeech2/model/mel_flow.py`

- [ ] Add `GlobalResponseNorm1D` operating on `[B, T, C]` tensors.
- [ ] Add `ConvNeXtV2Block1D` with depthwise `Conv1d`, `LayerNorm`, `Linear(C, expansion*C)`, `GELU`, `GRN`, `Linear(expansion*C, C)`, dropout, residual, and mask.
- [ ] Replace `FrameConditionSmoother` internals with stacked `ConvNeXtV2Block1D` blocks.

### Task 2: Update configs

**Files:**
- Modify: `FastSpeech2/config/AISHELL3/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech_paper/model.yaml`
- Modify: `FastSpeech2/config/LibriTTS/model.yaml`

- [ ] Set `conv_smoother_layers: 4`.
- [ ] Set `conv_smoother_kernel_size: 7`.
- [ ] Add `conv_smoother_expansion: 4`.

### Task 3: Verify and document

**Files:**
- Modify: `docs/no-stylecode-flow-mel-baseline-change-list.md`

- [ ] Run py_compile for FastSpeech2 Python files.
- [ ] Run `/tmp/flow_mel_smoke.py`.
- [ ] Update change list to mention ConvNeXt V2 smoother and latest verification output.
- [ ] Commit modified files.
