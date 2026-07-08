import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .mel_flow import SinusoidalTimeEmbedding, SinusoidalPositionEmbedding, MelFlowBlock
except ImportError:  # Support direct importlib loading in smoke tests.
    _mel_flow_path = Path(__file__).resolve().parent / "mel_flow.py"
    _spec = importlib.util.spec_from_file_location("mel_flow", _mel_flow_path)
    _mel_flow = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mel_flow)
    SinusoidalTimeEmbedding = _mel_flow.SinusoidalTimeEmbedding
    SinusoidalPositionEmbedding = _mel_flow.SinusoidalPositionEmbedding
    MelFlowBlock = _mel_flow.MelFlowBlock


def compute_V(x, y_pos, y_neg, temperature, mask_self=True):
    """Compute the drifting field using row/column balanced pairwise weights."""
    n = x.shape[0]
    n_pos = y_pos.shape[0]
    n_neg = y_neg.shape[0]

    dist_pos = torch.cdist(x, y_pos, p=2)
    dist_neg = torch.cdist(x, y_neg, p=2)

    if mask_self and n == n_neg:
        dist_neg = dist_neg + torch.eye(n, device=x.device, dtype=dist_neg.dtype) * 1e6

    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature
    logits = torch.cat([logit_pos, logit_neg], dim=1)

    weights_row = torch.softmax(logits, dim=1)
    weights_col = torch.softmax(logits, dim=0)
    weights = torch.sqrt(weights_row * weights_col)

    weights_pos = weights[:, :n_pos]
    weights_neg = weights[:, n_pos:]

    cross_pos = weights_pos * weights_neg.sum(dim=1, keepdim=True)
    cross_neg = weights_neg * weights_pos.sum(dim=1, keepdim=True)

    drift_pos = torch.mm(cross_pos, y_pos)
    drift_neg = torch.mm(cross_neg, y_neg)
    return drift_pos - drift_neg


def compute_V_multi_temperature(
    x,
    y_pos,
    y_neg,
    temperatures=[0.02, 0.05, 0.2],
    mask_self=True,
    normalize_each=True,
):
    """Sum drifting fields across temperatures, optionally RMS-normalizing each one."""
    v_total = torch.zeros_like(x)
    for temperature in temperatures:
        v_tau = compute_V(x, y_pos, y_neg, temperature, mask_self=mask_self)
        if normalize_each:
            v_tau = v_tau / torch.sqrt(torch.mean(v_tau**2) + 1e-8)
        v_total = v_total + v_tau
    return v_total


class MelDriftGenerator(nn.Module):
    def __init__(self, mel_dim, hidden_dim, config):
        super().__init__()
        config = config or {}
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.noise_scale = config.get("noise_scale", 1.0)
        self.alpha_min = config.get("alpha_min", 1.0)
        self.alpha_max = config.get("alpha_max", 3.0)
        self.temperatures = config.get("temperatures", [0.02, 0.05, 0.2])
        if isinstance(self.temperatures, (float, int)):
            self.temperatures = [float(self.temperatures)]
        self.normalize_each = config.get("normalize_each", True)
        self.max_tokens_per_utterance = config.get("max_tokens_per_utterance", 256)

        self.cfg_config = config.get("cfg", {})
        self.cfg_enabled = self.cfg_config.get("enabled", True)
        self.condition_dropout = self.cfg_config.get("condition_dropout", 0.1)
        self.default_guidance_scale = self.cfg_config.get("guidance_scale", 1.5)

        anchor_config = config.get("anchor", {})
        self.anchor_enabled = anchor_config.get("enabled", True)
        self.anchor_type = anchor_config.get("type", "l1")
        self.anchor_weight = anchor_config.get("weight", 0.05)
        if self.anchor_type not in {"l1", "mse"}:
            raise ValueError(f"Unsupported drift anchor loss type: {self.anchor_type}")

        alpha_dim = config.get("alpha_embed_dim", config.get("time_embed_dim", hidden_dim))
        depth = config.get("depth", 6)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        mlp_ratio = config.get("mlp_ratio", 4)
        max_seq_len = config.get("max_seq_len", 2000)

        self.input_proj = nn.Linear(mel_dim, hidden_dim)
        self.position = SinusoidalPositionEmbedding(hidden_dim, max_len=max_seq_len)
        self.alpha_embedding = SinusoidalTimeEmbedding(alpha_dim)
        self.alpha_to_hidden = nn.Linear(alpha_dim, alpha_dim)
        self.null_condition = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [MelFlowBlock(hidden_dim, alpha_dim, num_heads, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, mel_dim)

    def make_null_condition(self, condition):
        return self.null_condition.expand(condition.shape[0], condition.shape[1], -1)

    def maybe_drop_condition(self, condition, padding_mask=None):
        if not self.training or not self.cfg_enabled or self.condition_dropout <= 0:
            return condition
        drop = torch.rand(condition.shape[0], device=condition.device) < self.condition_dropout
        if not drop.any():
            return condition
        null_condition = self.make_null_condition(condition)
        condition = torch.where(drop[:, None, None], null_condition, condition)
        if padding_mask is not None:
            condition = condition.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return condition

    def _expand_alpha(self, alpha, batch_size, device, dtype):
        if alpha is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        if not torch.is_tensor(alpha):
            return torch.full((batch_size,), float(alpha), device=device, dtype=dtype)
        alpha = alpha.to(device=device, dtype=dtype)
        if alpha.dim() == 0 or alpha.numel() == 1:
            return alpha.reshape(1).expand(batch_size)
        alpha = alpha.reshape(-1)
        if alpha.shape[0] != batch_size:
            raise ValueError(f"alpha must have batch size {batch_size}, got {alpha.shape[0]}")
        return alpha

    def forward(
        self,
        x,
        alpha,
        frame_condition,
        padding_mask=None,
        force_uncond=False,
        apply_condition_dropout=True,
    ):
        if force_uncond:
            frame_condition = self.make_null_condition(frame_condition)
        elif apply_condition_dropout:
            frame_condition = self.maybe_drop_condition(frame_condition, padding_mask=padding_mask)

        if padding_mask is not None:
            frame_condition = frame_condition.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        alpha = self._expand_alpha(alpha, x.shape[0], x.device, x.dtype)
        alpha_emb = self.alpha_to_hidden(self.alpha_embedding(alpha))

        x = self.input_proj(x)
        x = self.position(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, frame_condition, alpha_emb, padding_mask=padding_mask)
        x = self.output_proj(self.output_norm(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def sample_noise(self, shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype) * self.noise_scale

    def _masked_targets(self, mel_targets, padding_mask):
        if padding_mask is None:
            return mel_targets
        return mel_targets.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _valid_mask(self, mel_targets, padding_mask):
        if padding_mask is None:
            return torch.ones(mel_targets.shape[:2], device=mel_targets.device, dtype=torch.bool)
        return ~padding_mask

    def _token_limit(self, valid_count):
        if self.max_tokens_per_utterance is None or self.max_tokens_per_utterance <= 0:
            return valid_count
        return min(valid_count, int(self.max_tokens_per_utterance))

    def _select_valid_indices(self, valid):
        indices = valid.nonzero(as_tuple=False).squeeze(-1)
        limit = self._token_limit(indices.shape[0])
        if limit < indices.shape[0]:
            if self.training:
                indices = indices[torch.randperm(indices.shape[0], device=indices.device)[:limit]]
            else:
                indices = indices[:limit]
        return indices

    def drift_loss(self, mel_pred, mel_targets, padding_mask=None):
        valid_mask = self._valid_mask(mel_targets, padding_mask)
        losses = []
        drift_norms = []
        zero = mel_pred.new_zeros(())

        for batch_idx in range(mel_pred.shape[0]):
            indices = self._select_valid_indices(valid_mask[batch_idx])
            if indices.shape[0] < 2:
                losses.append(zero)
                drift_norms.append(zero)
                continue

            feat_gen = F.normalize(mel_pred[batch_idx, indices], p=2, dim=-1, eps=1e-8)
            feat_pos = F.normalize(mel_targets[batch_idx, indices], p=2, dim=-1, eps=1e-8)
            v = compute_V_multi_temperature(
                feat_gen,
                feat_pos,
                feat_gen,
                temperatures=self.temperatures,
                mask_self=True,
                normalize_each=self.normalize_each,
            )
            target = (feat_gen + v).detach()
            losses.append(F.mse_loss(feat_gen, target))
            drift_norms.append(torch.sqrt(torch.mean(v**2) + 1e-8))

        if not losses:
            return zero, zero
        return torch.stack(losses).mean(), torch.stack(drift_norms).mean()

    def anchor_loss(self, mel_pred, mel_targets, padding_mask=None):
        if not self.anchor_enabled:
            return mel_pred.new_zeros(())
        valid_mask = self._valid_mask(mel_targets, padding_mask)
        if not valid_mask.any():
            return mel_pred.new_zeros(())

        pred = mel_pred[valid_mask]
        target = mel_targets[valid_mask]
        if self.anchor_type == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.l1_loss(pred, target)
        return loss * self.anchor_weight

    def training_step(self, mel_targets, frame_condition, padding_mask=None):
        mel_targets_masked = self._masked_targets(mel_targets, padding_mask)
        noise = self.sample_noise(mel_targets.shape, mel_targets.device, mel_targets.dtype)
        if padding_mask is not None:
            noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        alpha = torch.empty(mel_targets.shape[0], device=mel_targets.device, dtype=mel_targets.dtype).uniform_(
            self.alpha_min,
            self.alpha_max,
        )
        mel_pred = self.forward(noise, alpha, frame_condition, padding_mask=padding_mask)

        drift_loss, drift_norm = self.drift_loss(mel_pred, mel_targets_masked, padding_mask=padding_mask)
        anchor_loss = self.anchor_loss(mel_pred, mel_targets_masked, padding_mask=padding_mask)
        mel_loss = drift_loss + anchor_loss
        info = {
            "backend": "drift",
            "drift_loss": drift_loss,
            "anchor_loss": anchor_loss,
            "mel_loss": mel_loss,
            "sample_steps": 1,
            "drift_norm": drift_norm,
        }
        return mel_pred, mel_targets_masked, info

    def one_step_sample(self, frame_condition, padding_mask=None, guidance_scale=None, noise=None):
        guidance_scale = self.default_guidance_scale if guidance_scale is None else guidance_scale
        if noise is None:
            noise = self.sample_noise(
                (frame_condition.shape[0], frame_condition.shape[1], self.mel_dim),
                frame_condition.device,
                frame_condition.dtype,
            )
        if padding_mask is not None:
            noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        alpha = torch.full(
            (frame_condition.shape[0],),
            float(guidance_scale),
            device=frame_condition.device,
            dtype=frame_condition.dtype,
        )
        if self.cfg_enabled and guidance_scale != 1.0:
            mel_uncond = self.forward(
                noise,
                alpha,
                frame_condition,
                padding_mask=padding_mask,
                force_uncond=True,
                apply_condition_dropout=False,
            )
            mel_cond = self.forward(
                noise,
                alpha,
                frame_condition,
                padding_mask=padding_mask,
                force_uncond=False,
                apply_condition_dropout=False,
            )
            mel = mel_uncond + guidance_scale * (mel_cond - mel_uncond)
        else:
            mel = self.forward(
                noise,
                alpha,
                frame_condition,
                padding_mask=padding_mask,
                force_uncond=False,
                apply_condition_dropout=False,
            )
        if padding_mask is not None:
            mel = mel.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return mel

    def euler_sample(self, frame_condition, padding_mask=None, steps=32, guidance_scale=None):
        return self.one_step_sample(
            frame_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
        )


__all__ = ["compute_V", "compute_V_multi_temperature", "MelDriftGenerator"]
