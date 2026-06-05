import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.modulation = nn.Linear(time_dim, hidden_dim * 2)

    def forward(self, x, time_emb):
        scale, shift = self.modulation(time_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class FrameConditionSmoother(nn.Module):
    def __init__(self, hidden_dim, layers=2, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        blocks = []
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, padding_mask=None):
        residual = x
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out + residual)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class MelFlowBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim, num_heads, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.self_norm = AdaLayerNorm(hidden_dim, time_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = AdaLayerNorm(hidden_dim, time_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = AdaLayerNorm(hidden_dim, time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, condition, time_emb, padding_mask=None):
        h = self.self_norm(x, time_emb)
        h, _ = self.self_attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)
        x = x + h

        h = self.cross_norm(x, time_emb)
        h, _ = self.cross_attn(
            h,
            condition,
            condition,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + h

        x = x + self.mlp(self.mlp_norm(x, time_emb))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class MelFlowGenerator(nn.Module):
    def __init__(self, mel_dim, hidden_dim, config):
        super().__init__()
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.noise_scale = config.get("noise_scale", 1.0)
        self.cfg_config = config.get("cfg", {})
        self.cfg_enabled = self.cfg_config.get("enabled", True)
        self.condition_dropout = self.cfg_config.get("condition_dropout", 0.1)
        self.default_guidance_scale = self.cfg_config.get("guidance_scale", 1.5)

        time_dim = config.get("time_embed_dim", hidden_dim)
        depth = config.get("depth", 6)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        mlp_ratio = config.get("mlp_ratio", 4)
        max_seq_len = config.get("max_seq_len", 2000)

        self.input_proj = nn.Linear(mel_dim, hidden_dim)
        self.position = SinusoidalPositionEmbedding(hidden_dim, max_len=max_seq_len)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_to_hidden = nn.Linear(time_dim, time_dim)
        self.null_condition = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [MelFlowBlock(hidden_dim, time_dim, num_heads, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)]
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
        drop = drop[:, None, None]
        condition = torch.where(drop, null_condition, condition)
        if padding_mask is not None:
            condition = condition.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return condition

    def forward(self, x_t, t, frame_condition, padding_mask=None, force_uncond=False, apply_condition_dropout=True):
        if force_uncond:
            frame_condition = self.make_null_condition(frame_condition)
        elif apply_condition_dropout:
            frame_condition = self.maybe_drop_condition(frame_condition, padding_mask=padding_mask)

        if padding_mask is not None:
            frame_condition = frame_condition.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        time_emb = self.time_to_hidden(self.time_embedding(t))
        x = self.input_proj(x_t)
        x = self.position(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, frame_condition, time_emb, padding_mask=padding_mask)
        x = self.output_proj(self.output_norm(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def sample_noise(self, shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype) * self.noise_scale

    def training_step(self, mel_targets, frame_condition, padding_mask=None):
        x1 = mel_targets
        x0 = self.sample_noise(x1.shape, x1.device, x1.dtype)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        view_shape = (x1.shape[0],) + (1,) * (x1.dim() - 1)
        t_view = t.view(view_shape)
        x_t = (1 - t_view) * x0 + t_view * x1
        v_target = x1 - x0
        v_pred = self.forward(x_t, t, frame_condition, padding_mask=padding_mask)
        if padding_mask is not None:
            v_target = v_target.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return v_pred, v_target, {"t": t}

    def euler_sample(self, frame_condition, padding_mask=None, steps=32, guidance_scale=None):
        guidance_scale = self.default_guidance_scale if guidance_scale is None else guidance_scale
        z = self.sample_noise(
            (frame_condition.shape[0], frame_condition.shape[1], self.mel_dim),
            frame_condition.device,
            frame_condition.dtype,
        )
        if padding_mask is not None:
            z = z.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((frame_condition.shape[0],), i / steps, device=frame_condition.device, dtype=frame_condition.dtype)
            if self.cfg_enabled and guidance_scale != 1.0:
                v_uncond = self.forward(
                    z,
                    t,
                    frame_condition,
                    padding_mask=padding_mask,
                    force_uncond=True,
                    apply_condition_dropout=False,
                )
                v_cond = self.forward(
                    z,
                    t,
                    frame_condition,
                    padding_mask=padding_mask,
                    force_uncond=False,
                    apply_condition_dropout=False,
                )
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.forward(
                    z,
                    t,
                    frame_condition,
                    padding_mask=padding_mask,
                    force_uncond=False,
                    apply_condition_dropout=False,
                )
            z = z + dt * v
            if padding_mask is not None:
                z = z.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return z


def masked_flow_matching_loss(v_pred, v_target, padding_mask=None):
    if padding_mask is None:
        return F.mse_loss(v_pred, v_target)
    valid_mask = ~padding_mask
    if not valid_mask.any():
        return v_pred.new_zeros(())
    return F.mse_loss(v_pred[valid_mask], v_target[valid_mask])
