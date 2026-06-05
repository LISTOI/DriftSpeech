#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stylecode_predictor.data import StyleCodeJsonlDataset, make_collate_fn
from stylecode_predictor.flow_matching import euler_sample, flow_matching_loss
from stylecode_predictor.model import build_model_from_config


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_synthetic_jsonl(path, style_dim=32):
    records = [
        {
            "id": "sample_0",
            "speaker": 0,
            "text": "synthetic 0",
            "src_len": 4,
            "mel_len": 12,
            "bottleneck_dim": style_dim,
            "durations": [3, 2, 0, 7],
            "stylecode_shape": [4, style_dim],
            "stylecode": torch.randn(4, style_dim).tolist(),
            "token_ids": [1, 2, 3, 4],
            "phoneme_text": "AA BB CC DD",
        },
        {
            "id": "sample_1",
            "speaker": 0,
            "text": "synthetic 1",
            "src_len": 2,
            "mel_len": 5,
            "bottleneck_dim": style_dim,
            "durations": [1, 4],
            "stylecode_shape": [2, style_dim],
            "stylecode": torch.randn(2, style_dim).tolist(),
            "token_ids": [2, 5],
            "phoneme_text": "BB EE",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)
    config = load_config(args.config)
    config["model"]["vocab_size"] = 16
    config["model"]["style_dim"] = 32
    config["model"]["max_seq_len"] = 16
    config["model"]["text_encoder"]["hidden_dim"] = 64
    config["model"]["text_encoder"]["num_layers"] = 2
    config["model"]["text_encoder"]["num_heads"] = 2
    config["model"]["text_encoder"]["ffn_dim"] = 128
    config["model"]["dit"]["hidden_dim"] = 64
    config["model"]["dit"]["depth"] = 2
    config["model"]["dit"]["num_heads"] = 2
    config["model"]["dit"]["time_embed_dim"] = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "synthetic.jsonl")
        make_synthetic_jsonl(jsonl_path, style_dim=32)
        dataset = StyleCodeJsonlDataset([jsonl_path])
        loader = DataLoader(dataset, batch_size=2, collate_fn=make_collate_fn(0))
        batch = next(iter(loader))
        assert batch["tokens"].shape == (2, 4)
        assert batch["stylecode"].shape == (2, 4, 32)
        assert batch["padding_mask"].shape == (2, 4)

        model = build_model_from_config(config)
        loss, info = flow_matching_loss(
            model,
            batch["tokens"],
            batch["stylecode"],
            batch["lengths"],
            batch["padding_mask"],
            durations=batch["durations"],
        )
        assert torch.isfinite(loss).item()
        assert info["v_pred"].shape == batch["stylecode"].shape
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss.backward()
        optimizer.step()
        pred = euler_sample(
            model,
            batch["tokens"],
            batch["lengths"],
            batch["padding_mask"],
            style_dim=32,
            durations=batch["durations"],
            num_steps=4,
        )
        assert pred.shape == batch["stylecode"].shape
        assert torch.isfinite(pred).all().item()
        assert pred[batch["padding_mask"]].abs().max().item() == 0.0
    print("smoke test ok")


if __name__ == "__main__":
    main()
