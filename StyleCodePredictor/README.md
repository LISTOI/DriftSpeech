# StyleCodePredictor

Standalone flow-matching predictor for FastSpeech2 phoneme-level style codes.

## Data

Training data is read from FastSpeech2 exported stylecode JSONL files, for example:

```text
/home/listoi/Speech/FastSpeech2/output/result/LJSpeech/stylecode/900000.jsonl
```

Each record should contain `token_ids`, `stylecode`, `durations`, and `src_len`.

## Install

```bash
conda run -n torch_gpu pip install -r /home/listoi/Speech/StyleCodePredictor/requirements.txt
```

## Validate JSONL

```bash
conda run -n torch_gpu python /home/listoi/Speech/StyleCodePredictor/scripts/validate_jsonl.py \
  --jsonl /home/listoi/Speech/FastSpeech2/output/result/LJSpeech/stylecode/900000.jsonl \
  --require-token-ids
```

## Smoke test

```bash
conda run -n torch_gpu python /home/listoi/Speech/StyleCodePredictor/scripts/smoke_test.py \
  --config /home/listoi/Speech/StyleCodePredictor/configs/default.yaml
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 nohup conda run -n torch_gpu python /home/listoi/Speech/StyleCodePredictor/scripts/train.py \
  --config /home/listoi/Speech/StyleCodePredictor/configs/default.yaml \
  > /home/listoi/Speech/StyleCodePredictor/train.log 2>&1 &
```

For a short sanity check:

```bash
conda run -n torch_gpu python /home/listoi/Speech/StyleCodePredictor/scripts/train.py \
  --config /home/listoi/Speech/StyleCodePredictor/configs/default.yaml \
  --max_steps 20
```

## Infer

```bash
conda run -n torch_gpu python /home/listoi/Speech/StyleCodePredictor/scripts/infer.py \
  --checkpoint /home/listoi/Speech/StyleCodePredictor/runs/default/checkpoints/latest.pt \
  --jsonl /home/listoi/Speech/FastSpeech2/output/result/LJSpeech/stylecode/900000.jsonl \
  --output_jsonl /home/listoi/Speech/StyleCodePredictor/runs/default/predicted_stylecodes.jsonl \
  --num_steps 32
```
