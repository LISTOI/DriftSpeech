import os

import torch


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config,
        "style_mean": style_mean.detach().cpu() if style_mean is not None else None,
        "style_std": style_std.detach().cpu() if style_std is not None else None,
        "vocab_size": vocab_size,
        "style_dim": style_dim,
    }
    torch.save(checkpoint, path)


def save_latest_and_step(output_dir, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    step_path = os.path.join(ckpt_dir, f"step_{step}.pt")
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    save_checkpoint(step_path, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim)
    save_checkpoint(latest_path, model, optimizer, step, epoch, config, style_mean, style_std, vocab_size, style_dim)
    return latest_path
