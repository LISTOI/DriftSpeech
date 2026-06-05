import torch
from torch import nn

from .masks import mask_sequence
from .positional import SinusoidalTimeEmbedding


class TimeMLP(nn.Module):
    def __init__(self, time_embed_dim, hidden_dim):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t):
        return self.net(self.time_embedding(t))


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, x, time_emb):
        scale, shift = self.modulation(time_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class SeqDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.self_norm = AdaLayerNorm(hidden_dim)
        self.cross_norm = AdaLayerNorm(hidden_dim)
        self.mlp_norm = AdaLayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_cond, time_emb, padding_mask):
        residual = x
        x_norm = self.self_norm(x, time_emb)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)
        x = mask_sequence(x, padding_mask)

        residual = x
        x_norm = self.cross_norm(x, time_emb)
        cross_out, _ = self.cross_attn(
            x_norm,
            text_cond,
            text_cond,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(cross_out)
        x = mask_sequence(x, padding_mask)

        residual = x
        x = residual + self.mlp(self.mlp_norm(x, time_emb))
        return mask_sequence(x, padding_mask)


class SeqDiT(nn.Module):
    def __init__(self, hidden_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [SeqDiTBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, text_cond, time_emb, padding_mask):
        for layer in self.layers:
            x = layer(x, text_cond, time_emb, padding_mask)
        x = self.final_norm(x)
        return mask_sequence(x, padding_mask)
