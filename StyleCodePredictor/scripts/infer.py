#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stylecode_predictor.checkpoint import load_checkpoint
from stylecode_predictor.data import StyleCodeJsonlDataset, make_collate_fn
from stylecode_predictor.flow_matching import euler_sample
from stylecode_predictor.model import build_model_from_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    result = dict(batch)
    for key in ["tokens", "stylecode", "durations", "lengths", "padding_mask"]:
        result[key] = batch[key].to(device)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    config = ckpt["config"]
    data_config = config.get("data", {})
    model = build_model_from_config(
        config,
        vocab_size=int(ckpt.get("vocab_size", config["model"]["vocab_size"])),
        style_dim=int(ckpt.get("style_dim", config["model"]["style_dim"])),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    style_mean = ckpt.get("style_mean")
    style_std = ckpt.get("style_std")
    if style_mean is not None:
        style_mean = style_mean.to(device)
        style_std = style_std.to(device)

    dataset = StyleCodeJsonlDataset(
        args.jsonl,
        require_token_ids=bool(data_config.get("require_token_ids", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(int(data_config.get("pad_token_id", 0))),
    )
    num_steps = args.num_steps or int(config.get("inference", {}).get("num_steps", 32))
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode(), open(args.output_jsonl, "w", encoding="utf-8") as f:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = euler_sample(
                model,
                batch["tokens"],
                batch["lengths"],
                batch["padding_mask"],
                style_dim=int(ckpt.get("style_dim", config["model"]["style_dim"])),
                durations=batch["durations"],
                num_steps=num_steps,
                noise_scale=float(config.get("flow", {}).get("noise_scale", 1.0)),
            )
            if style_mean is not None:
                pred = pred * style_std[None, None, :] + style_mean[None, None, :]
            pred = pred.cpu()
            lengths = batch["lengths"].cpu().tolist()
            tokens = batch["tokens"].cpu().tolist()
            for i, length in enumerate(lengths):
                length = int(length)
                style = pred[i, :length].tolist()
                record = {
                    "id": batch["ids"][i],
                    "speaker": batch["speakers"][i],
                    "text": batch["texts"][i],
                    "src_len": length,
                    "token_ids": tokens[i][:length],
                    "phoneme_text": batch["phoneme_texts"][i],
                    "predicted_stylecode_shape": [length, len(style[0]) if style else 0],
                    "predicted_stylecode": style,
                    "checkpoint": args.checkpoint,
                    "num_steps": num_steps,
                    "seed": args.seed,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
