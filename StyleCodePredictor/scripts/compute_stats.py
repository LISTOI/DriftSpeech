#!/usr/bin/env python3
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stylecode_predictor.data import StyleCodeJsonlDataset, make_collate_fn
from stylecode_predictor.masks import make_valid_mask


def compute_style_stats(jsonl_paths, batch_size, pad_token_id, mask_zero_duration, require_token_ids):
    dataset = StyleCodeJsonlDataset(jsonl_paths, require_token_ids=require_token_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(pad_token_id),
    )
    style_dim = dataset.style_dim
    total = torch.zeros(style_dim)
    total_sq = torch.zeros(style_dim)
    count = 0
    for batch in loader:
        valid_mask = make_valid_mask(
            batch["lengths"],
            durations=batch["durations"],
            mask_zero_duration=mask_zero_duration,
            max_len=batch["stylecode"].shape[1],
        )
        values = batch["stylecode"][valid_mask]
        if values.numel() == 0:
            continue
        total += values.sum(dim=0)
        total_sq += values.pow(2).sum(dim=0)
        count += values.shape[0]
    if count == 0:
        raise RuntimeError("No valid stylecode positions found")
    mean = total / count
    var = (total_sq / count - mean.pow(2)).clamp_min(1e-12)
    std = var.sqrt().clamp_min(1e-6)
    return mean, std, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--mask-zero-duration", action="store_true")
    parser.add_argument("--require-token-ids", action="store_true")
    args = parser.parse_args()

    mean, std, count = compute_style_stats(
        args.jsonl,
        args.batch_size,
        args.pad_token_id,
        args.mask_zero_duration,
        args.require_token_ids,
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"mean": mean, "std": std, "count": count}, args.output)
    print(f"Saved stats to {args.output} with count={count}")


if __name__ == "__main__":
    main()
