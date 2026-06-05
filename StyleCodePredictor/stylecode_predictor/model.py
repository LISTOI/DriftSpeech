import torch
from torch import nn

from .dit import SeqDiT, TimeMLP
from .masks import mask_sequence
from .positional import SinusoidalPositionalEncoding
from .text_encoder import TokenTextEncoder


class StyleCodeFlowPredictor(nn.Module):
    def __init__(
        self,
        vocab_size,
        style_dim,
        max_seq_len,
        text_encoder_config,
        dit_config,
        pad_token_id=0,
    ):
        super().__init__()
        hidden_dim = int(dit_config.get("hidden_dim", text_encoder_config.get("hidden_dim", 256)))
        text_hidden_dim = int(text_encoder_config.get("hidden_dim", hidden_dim))
        if text_hidden_dim != hidden_dim:
            raise ValueError("text_encoder.hidden_dim must match dit.hidden_dim in the initial implementation")

        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.text_encoder = TokenTextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=int(text_encoder_config.get("num_layers", 4)),
            num_heads=int(text_encoder_config.get("num_heads", 2)),
            ffn_dim=int(text_encoder_config.get("ffn_dim", hidden_dim * 4)),
            conv_kernel_size=int(text_encoder_config.get("conv_kernel_size", 9)),
            dropout=float(text_encoder_config.get("dropout", 0.2)),
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
        )
        self.style_in = nn.Linear(style_dim, hidden_dim)
        self.style_position = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)
        self.time_mlp = TimeMLP(int(dit_config.get("time_embed_dim", hidden_dim)), hidden_dim)
        self.dit = SeqDiT(
            hidden_dim=hidden_dim,
            depth=int(dit_config.get("depth", 8)),
            num_heads=int(dit_config.get("num_heads", 4)),
            mlp_ratio=float(dit_config.get("mlp_ratio", 4)),
            dropout=float(dit_config.get("dropout", 0.1)),
        )
        self.style_out = nn.Linear(hidden_dim, style_dim)

    def forward(self, tokens, x_t, t, padding_mask, durations=None):
        del durations
        if t.dim() == 0:
            t = t.expand(tokens.shape[0])
        text_cond = self.text_encoder(tokens, padding_mask)
        style_hidden = self.style_position(self.style_in(x_t))
        style_hidden = mask_sequence(style_hidden, padding_mask)
        time_emb = self.time_mlp(t.to(dtype=x_t.dtype))
        style_hidden = self.dit(style_hidden, text_cond, time_emb, padding_mask)
        v_pred = self.style_out(style_hidden)
        return mask_sequence(v_pred, padding_mask)


def build_model_from_config(config, vocab_size=None, style_dim=None):
    data_config = config.get("data", {})
    model_config = config["model"]
    return StyleCodeFlowPredictor(
        vocab_size=int(vocab_size or model_config["vocab_size"]),
        style_dim=int(style_dim or model_config["style_dim"]),
        max_seq_len=int(model_config.get("max_seq_len", 1000)),
        text_encoder_config=model_config.get("text_encoder", {}),
        dit_config=model_config.get("dit", {}),
        pad_token_id=int(data_config.get("pad_token_id", 0)),
    )
