#!/usr/bin/env python3
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stylecode_predictor.checkpoint import load_checkpoint, save_latest_and_step
from stylecode_predictor.data import StyleCodeJsonlDataset, make_collate_fn
from stylecode_predictor.flow_matching import euler_sample, flow_matching_loss
from stylecode_predictor.masks import make_valid_mask
from stylecode_predictor.metrics import masked_cosine_similarity, masked_mae, masked_mse
from stylecode_predictor.model import build_model_from_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch, device):
    result = dict(batch)
    for key in ["tokens", "stylecode", "durations", "lengths", "padding_mask"]:
        result[key] = batch[key].to(device)
    return result


def compute_style_stats(dataset, pad_token_id, mask_zero_duration):
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate_fn(pad_token_id),
    )
    total = torch.zeros(dataset.style_dim)
    total_sq = torch.zeros(dataset.style_dim)
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
    return mean, std


def normalize_batch_stylecode(batch, style_mean, style_std):
    if style_mean is None or style_std is None:
        return batch
    batch = dict(batch)
    batch["stylecode"] = (batch["stylecode"] - style_mean[None, None, :]) / style_std[None, None, :]
    return batch


@torch.no_grad()
def evaluate(model, loader, config, device, style_mean, style_std, max_batches=4):
    model.eval()
    data_config = config.get("data", {})
    flow_config = config.get("flow", {})
    style_dim = int(config["model"]["style_dim"])
    totals = {
        "flow_loss": 0.0,
        "sample_mse": 0.0,
        "sample_mae": 0.0,
        "sample_cosine": 0.0,
    }
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        batch = normalize_batch_stylecode(batch, style_mean, style_std)
        loss, _ = flow_matching_loss(
            model,
            batch["tokens"],
            batch["stylecode"],
            batch["lengths"],
            batch["padding_mask"],
            durations=batch["durations"],
            mask_zero_duration=bool(data_config.get("mask_zero_duration", True)),
            t_min=float(flow_config.get("t_min", 0.0)),
            t_max=float(flow_config.get("t_max", 1.0)),
            noise_scale=float(flow_config.get("noise_scale", 1.0)),
        )
        pred = euler_sample(
            model,
            batch["tokens"],
            batch["lengths"],
            batch["padding_mask"],
            style_dim=style_dim,
            durations=batch["durations"],
            num_steps=int(config.get("inference", {}).get("num_steps", 32)),
            noise_scale=float(flow_config.get("noise_scale", 1.0)),
        )
        valid_mask = make_valid_mask(
            batch["lengths"],
            durations=batch["durations"],
            mask_zero_duration=bool(data_config.get("mask_zero_duration", True)),
            max_len=batch["stylecode"].shape[1],
        )
        totals["flow_loss"] += float(loss.item())
        totals["sample_mse"] += float(masked_mse(pred, batch["stylecode"], valid_mask).item())
        totals["sample_mae"] += float(masked_mae(pred, batch["stylecode"], valid_mask).item())
        totals["sample_cosine"] += float(masked_cosine_similarity(pred, batch["stylecode"], valid_mask).item())
        count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    flow_config = config.get("flow", {})
    set_seed(int(train_config.get("seed", 1234)))

    train_dataset = StyleCodeJsonlDataset(
        data_config["train_jsonl"],
        require_token_ids=bool(data_config.get("require_token_ids", True)),
    )
    val_paths = data_config.get("val_jsonl") or []
    val_dataset = StyleCodeJsonlDataset(
        val_paths,
        require_token_ids=bool(data_config.get("require_token_ids", True)),
    ) if val_paths else None

    pad_token_id = int(data_config.get("pad_token_id", 0))
    vocab_size = max(int(config["model"].get("vocab_size", 0)), train_dataset.max_token_id + 1)
    style_dim = int(train_dataset.style_dim)
    config["model"]["vocab_size"] = vocab_size
    config["model"]["style_dim"] = style_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 2)),
        collate_fn=make_collate_fn(pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_config.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_config.get("num_workers", 2)),
            collate_fn=make_collate_fn(pad_token_id),
            pin_memory=torch.cuda.is_available(),
        )

    style_mean = None
    style_std = None
    if bool(data_config.get("normalize_stylecodes", True)):
        stats_path = data_config.get("stats_path")
        if stats_path:
            stats = load_checkpoint(stats_path, map_location="cpu")
            style_mean = stats["mean"].float()
            style_std = stats["std"].float()
        else:
            style_mean, style_std = compute_style_stats(
                train_dataset,
                pad_token_id,
                bool(data_config.get("mask_zero_duration", True)),
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if style_mean is not None:
        style_mean = style_mean.to(device)
        style_std = style_std.to(device)

    model = build_model_from_config(config, vocab_size=vocab_size, style_dim=style_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_config.get("amp", True)) and device.type == "cuda")
    start_epoch = 0
    step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))

    output_dir = train_config.get("output_dir", "/home/listoi/Speech/StyleCodePredictor/runs/default")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

    model.train()
    max_steps = args.max_steps
    epochs = int(train_config.get("epochs", 200))
    progress = tqdm(total=max_steps, desc="training") if max_steps else None
    try:
        for epoch in range(start_epoch, epochs):
            for batch in train_loader:
                step += 1
                batch = move_batch_to_device(batch, device)
                batch = normalize_batch_stylecode(batch, style_mean, style_std)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(train_config.get("amp", True)) and device.type == "cuda"):
                    loss, _ = flow_matching_loss(
                        model,
                        batch["tokens"],
                        batch["stylecode"],
                        batch["lengths"],
                        batch["padding_mask"],
                        durations=batch["durations"],
                        mask_zero_duration=bool(data_config.get("mask_zero_duration", True)),
                        t_min=float(flow_config.get("t_min", 0.0)),
                        t_max=float(flow_config.get("t_max", 1.0)),
                        noise_scale=float(flow_config.get("noise_scale", 1.0)),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()

                if step % int(train_config.get("log_interval", 50)) == 0 or step == 1:
                    writer.add_scalar("train/flow_loss", loss.item(), step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                    print(f"step={step} epoch={epoch} flow_loss={loss.item():.6f}")

                if val_loader is not None and step % int(train_config.get("val_interval", 1000)) == 0:
                    metrics = evaluate(model, val_loader, config, device, style_mean, style_std)
                    for key, value in metrics.items():
                        writer.add_scalar(f"val/{key}", value, step)
                    print("validation", {key: round(value, 6) for key, value in metrics.items()})

                if step % int(train_config.get("save_interval", 5000)) == 0:
                    save_latest_and_step(output_dir, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim)

                if progress is not None:
                    progress.update(1)
                if max_steps is not None and step >= max_steps:
                    save_latest_and_step(output_dir, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim)
                    return
        save_latest_and_step(output_dir, model, optimizer, step, epochs, config, style_mean, style_std, vocab_size, style_dim)
    finally:
        if progress is not None:
            progress.close()
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
