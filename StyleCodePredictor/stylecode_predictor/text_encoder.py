import torch
from torch import nn

from .masks import mask_sequence
from .positional import SinusoidalPositionalEncoding


class FFTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, conv_kernel_size, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        padding = (conv_kernel_size - 1) // 2
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_dim, ffn_dim, conv_kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_dim, hidden_dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask):
        residual = x
        x_norm = self.attn_norm(x)
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
        x_norm = self.ffn_norm(x).transpose(1, 2)
        ffn_out = self.ffn(x_norm).transpose(1, 2)
        x = residual + ffn_out
        return mask_sequence(x, padding_mask)


class TokenTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        ffn_dim,
        conv_kernel_size,
        dropout,
        max_seq_len,
        pad_token_id=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                FFTBlock(hidden_dim, num_heads, ffn_dim, conv_kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, padding_mask):
        x = self.embedding(tokens)
        x = self.position(x)
        x = self.dropout(x)
        x = mask_sequence(x, padding_mask)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)
        return mask_sequence(x, padding_mask)
