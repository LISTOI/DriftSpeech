# DriftSpeech

DriftSpeech is an experimental speech synthesis workspace based on a refactored FastSpeech2 pipeline. The current focus is phoneme-level style code extraction, semantic leakage evaluation, adversarial/VQ regularization, and flow-matching style or acoustic prediction experiments.

## Repository layout

```text
.
├── FastSpeech2/             # Refactored FastSpeech2 training, synthesis, and style extractor code
├── StyleCodePredictor/      # Standalone flow-matching predictor for phoneme-level style codes
├── evaluate/                # Style code visualization and leakage/codebook evaluation tools
└── README.md
```

## Main components

### FastSpeech2

The FastSpeech2 branch in this repository has been adapted for explicit-duration speech synthesis experiments:

- pitch and energy prediction code has been removed;
- duration prediction and duration loss are retained;
- mel targets can be encoded into phoneme-level style codes through duration pooling;
- the style extractor supports optional phoneme-adversarial training;
- the style extractor supports optional vector quantization (VQ);
- checkpoint-time style code export is used for downstream leakage analysis.

Typical entry points:

```bash
cd FastSpeech2
conda run -n torch_gpu python train.py \
  -p config/LJSpeech/preprocess.yaml \
  -m config/LJSpeech/model.yaml \
  -t config/LJSpeech/train.yaml
```

### StyleCodePredictor

`StyleCodePredictor` is a standalone PyTorch project for predicting phoneme-level style codes from text/token IDs with a flow-matching model.

High-level flow:

```text
token_ids -> text encoder -> text condition
Gaussian noise + time + text condition -> sequence DiT / cross-attention -> predicted style code
```

Useful commands:

```bash
conda run -n torch_gpu python StyleCodePredictor/scripts/smoke_test.py \
  --config StyleCodePredictor/configs/default.yaml

conda run -n torch_gpu python StyleCodePredictor/scripts/train.py \
  --config StyleCodePredictor/configs/default.yaml
```

### Evaluation tools

The `evaluate` directory contains tools for phoneme-level style code analysis:

- t-SNE projection and phoneme coloring;
- kNN same-label accuracy in original style-code space;
- silhouette metrics;
- VQ code usage, perplexity, purity, and NMI metrics;
- multi-seed and multi-perplexity batch evaluation.

Example:

```bash
conda run -n torch_gpu python evaluate/visualize_phoneme_stylecodes.py \
  --jsonl FastSpeech2/output/result/LJSpeech/stylecode/900000.jsonl \
  --output_dir FastSpeech2/output/result/LJSpeech/stylecode_eval/900000 \
  --max_points 5000 \
  --top_k 20 \
  --perplexity 30 \
  --seed 1234
```

## Data and generated files

Large or machine-local files are intentionally not tracked by Git, including:

- raw datasets;
- preprocessed data;
- model checkpoints;
- vocoder weights;
- training logs;
- generated audio, figures, and evaluation outputs.

See `.gitignore` for the exact ignore rules. If a command expects local data or checkpoints, place them under the expected local paths before running.

## Environment

The code is intended to run in a Conda environment named `torch_gpu`.

Install project dependencies as needed:

```bash
conda run -n torch_gpu pip install -r FastSpeech2/requirements.txt
conda run -n torch_gpu pip install -r StyleCodePredictor/requirements.txt
```

For evaluation plots and metrics, make sure common scientific Python packages are available:

```bash
conda run -n torch_gpu pip install numpy torch scikit-learn matplotlib seaborn librosa umap-learn
```

## Suggested branch workflow

Use `main` as the stable baseline and create one branch per experiment:

```bash
git checkout main
git pull
git checkout -b exp/adversarial-only
```

Suggested branch names:

```text
exp/adversarial-only
exp/vq-only
exp/vq-adversarial
exp/flow-stylecode-predictor
exp/flow-mel-baseline
```

When starting a new experiment, keep config/output paths distinct so results do not overwrite each other.

## Current research direction

This repository is being used to compare ways of representing and predicting phoneme-level style information while reducing phoneme or semantic leakage:

1. baseline phoneme-level style extraction;
2. adversarial suppression of phoneme identity in style codes;
3. VQ bottleneck/codebook regularization;
4. t-SNE/kNN/silhouette/codebook metrics for leakage analysis;
5. flow-matching predictors for style codes and future mel-spectrogram baselines.
