# Flow-Matching with StyleCode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add phoneme-level stylecode conditioning back into the flow-matching mel baseline.

**Architecture:** The model extracts phoneme-level stylecode from GT/reference mel during training/evaluation, decodes it to hidden size, fuses it with text encoder hidden states, expands the fused condition to frame level with duration targets, and feeds the ConvNeXt V2-smoothed frame condition into the mel flow generator. Losses include flow mel loss, duration loss, optional adversarial phoneme loss, and optional VQ loss.

**Tech Stack:** PyTorch, existing FastSpeech2 model modules, existing `PhonemeStyleExtractor`, existing flow-mel baseline.

---

## File Structure

- Modify `FastSpeech2/model/fastspeech2.py`: restore `PhonemeStyleExtractor`, style-hidden fusion, style info/logits returns, zero-style fallback.
- Modify `FastSpeech2/model/loss.py`: restore adversarial and VQ losses around flow mel loss.
- Modify `FastSpeech2/train.py`: restore stylecode dump support for stylecode branch and update loss log formatting.
- Modify `FastSpeech2/evaluate.py`: update validation log formatting.
- Modify `FastSpeech2/utils/tools.py`: update TensorBoard scalar tags.
- Modify `FastSpeech2/evaluate_flow_mel.py`: keep GT-duration sampling path and allow GT/reference mel to be used only for style extraction.
- Modify `FastSpeech2/config/*/model.yaml`: set adversarial recommended defaults for this branch.
- Modify `FastSpeech2/config/*/train.yaml`: set stylecode logging enabled and recommend branch-specific output paths for LJSpeech.

## Tasks

### Task 1: Restore stylecode conditioning in model forward

**Files:**
- Modify: `FastSpeech2/model/fastspeech2.py`

- [ ] Import `PhonemeStyleExtractor`.
- [ ] Instantiate `self.style_extractor`.
- [ ] When `mels` and `d_targets` are present, extract `stylecode, style_info` and decode to `style_hidden`.
- [ ] Fuse with `phoneme_condition = text_hidden + style_hidden` before length regulation.
- [ ] Return `style_adv_logits` and `style_info` in the output tuple after `flow_info`.
- [ ] Use zero style hidden when no mel/reference is available.

### Task 2: Restore style-aware losses

**Files:**
- Modify: `FastSpeech2/model/loss.py`

- [ ] Read `style_adv_logits` and `style_info` from the extended prediction tuple.
- [ ] Compute `phoneme_adv_loss` on valid phoneme positions when enabled.
- [ ] Add optional VQ losses from `style_info`.
- [ ] Return `(total_loss, flow_mel_loss, duration_loss, phoneme_adv_loss, vq_loss, vq_commitment_loss, vq_codebook_loss, codebook_perplexity, used_codes)` when style_info exists; otherwise return through adversarial loss.

### Task 3: Update logs and scalar tags

**Files:**
- Modify: `FastSpeech2/train.py`
- Modify: `FastSpeech2/evaluate.py`
- Modify: `FastSpeech2/utils/tools.py`

- [ ] Log `Total Loss`, `Flow Mel Loss`, `Duration Loss`, `Phoneme Adv Loss`, and VQ metrics when present.
- [ ] Allow `dump_stylecodes()` to run again on this branch because `style_extractor` is restored.
- [ ] Keep mel/audio sample logging compatible with prediction tuple index `predictions[7]` for mel lengths.

### Task 4: Update GT-style flow evaluation

**Files:**
- Modify: `FastSpeech2/evaluate_flow_mel.py`

- [ ] Ensure `sample_with_gt_duration()` passes `mels=batch[6]`, `mel_lens=batch[7]`, and `d_targets=batch[9]` so the model can extract stylecode.
- [ ] Ensure generated mel is sampled from flow and GT mel is used only as style/reference input and metric target.

### Task 5: Update configs

**Files:**
- Modify: `FastSpeech2/config/AISHELL3/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech_paper/model.yaml`
- Modify: `FastSpeech2/config/LibriTTS/model.yaml`
- Modify: `FastSpeech2/config/LJSpeech/train.yaml`

- [ ] Set `phoneme_style.adversarial.enabled: true`.
- [ ] Set `phoneme_style.adversarial.weight: 0.02`.
- [ ] Set `phoneme_style.adversarial.grl_lambda: 0.5`.
- [ ] Keep `phoneme_style.vq.enabled: false`.
- [ ] Enable `stylecode_logging.enabled: true`.
- [ ] Set LJSpeech output paths to `LJSpeech_flow_stylecode_adv`.

### Task 6: Verify

**Files:**
- Test modified Python files.

- [ ] Run py_compile on FastSpeech2 model/train/evaluate/synthesize/utils/evaluate_flow_mel files.
- [ ] Run an isolated smoke test with mocked transformer/text dependencies to verify forward, loss, backward, sampling, adversarial loss tuple, and GT-style evaluation sampling path.
- [ ] Commit implementation changes.
