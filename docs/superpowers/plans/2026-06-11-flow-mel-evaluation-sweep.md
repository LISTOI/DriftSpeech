# Flow Mel Evaluation Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GT-duration evaluation sweep script for comparing flow-mel checkpoints, CFG guidance scales, and Euler sample step counts.

**Architecture:** Create a standalone FastSpeech2 script that loads validation samples with GT mel and GT duration, samples mel from the flow model without passing mel targets, computes frame-aligned mel metrics, optionally saves generated/GT wav and mel plots, and writes CSV/JSON summaries plus the best automatic candidate.

**Tech Stack:** PyTorch, NumPy, YAML, existing FastSpeech2 `Dataset`, `get_model`, `get_vocoder`, `vocoder_infer`, and `plot_mel` helpers.

---

## File Structure

- Create `FastSpeech2/evaluate_flow_mel.py`: checkpoint/cfg/sample-step sweep, GT-duration sampling, metrics, artifact writing.
- Modify `docs/no-stylecode-flow-mel-baseline-change-list.md`: document the new evaluation script and intended metrics.

## Tasks

### Task 1: Implement evaluation helper functions

**Files:**
- Create: `FastSpeech2/evaluate_flow_mel.py`

- [ ] Add parsing helpers for comma-separated int/float lists.
- [ ] Add `set_seed(seed)` for deterministic sampling.
- [ ] Add `metric_summary(values)` returning mean/std.
- [ ] Add `compute_mel_metrics(pred, target, length)` for L1/MSE/RMSE/mean-diff/std-diff.
- [ ] Add `score_row(row)` using `mel_l1_mean + 0.1 * mel_std_abs_diff_mean`.

### Task 2: Implement GT-duration sampling and artifact saving

**Files:**
- Create: `FastSpeech2/evaluate_flow_mel.py`

- [ ] Load `Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)`.
- [ ] For each batch, call model with `mels=None` and `d_targets=batch[9]` so GT duration drives generation.
- [ ] Compute metrics against `batch[6]` GT mel up to `mel_len`.
- [ ] Save first `save_samples` generated mel PNG/wav and GT mel PNG/wav when vocoder is enabled.

### Task 3: Implement sweep outputs

**Files:**
- Create: `FastSpeech2/evaluate_flow_mel.py`

- [ ] Loop over `restore_steps × guidance_scales × sample_steps`.
- [ ] Override `train_config["mel_flow"]["guidance_scale"]` and `sample_steps` for each run.
- [ ] Write `summary.csv`, `summary.json`, `best.json`, and per-combination sample folders.
- [ ] Print the best automatic combination.

### Task 4: Verify and document

**Files:**
- Modify: `docs/no-stylecode-flow-mel-baseline-change-list.md`

- [ ] Run `conda run -n torch_gpu python -m py_compile FastSpeech2/evaluate_flow_mel.py`.
- [ ] Run a helper-only smoke test if no local checkpoints/data are available.
- [ ] Update the change list with the evaluation script and metric list.
- [ ] Commit the script, plan, and change-list update.
