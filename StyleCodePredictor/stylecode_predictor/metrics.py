import torch
import torch.nn.functional as F


def _expand_valid_mask(valid_mask, x):
    while valid_mask.dim() < x.dim():
        valid_mask = valid_mask.unsqueeze(-1)
    return valid_mask


def masked_mse(pred, target, valid_mask):
    valid = _expand_valid_mask(valid_mask, pred)
    diff = (pred - target).pow(2).masked_select(valid)
    if diff.numel() == 0:
        return pred.new_zeros(())
    return diff.mean()


def masked_mae(pred, target, valid_mask):
    valid = _expand_valid_mask(valid_mask, pred)
    diff = (pred - target).abs().masked_select(valid)
    if diff.numel() == 0:
        return pred.new_zeros(())
    return diff.mean()


def masked_cosine_similarity(pred, target, valid_mask):
    valid = valid_mask.bool()
    if valid.sum().item() == 0:
        return pred.new_zeros(())
    pred_flat = pred[valid]
    target_flat = target[valid]
    return F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()


def style_stats_distance(pred, target, valid_mask):
    valid = valid_mask.bool()
    if valid.sum().item() == 0:
        return pred.new_zeros(())
    pred_flat = pred[valid]
    target_flat = target[valid]
    pred_mean = pred_flat.mean(dim=0)
    target_mean = target_flat.mean(dim=0)
    pred_std = pred_flat.std(dim=0, unbiased=False)
    target_std = target_flat.std(dim=0, unbiased=False)
    return (pred_mean - target_mean).pow(2).mean().sqrt() + (pred_std - target_std).pow(2).mean().sqrt()
