from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Conv
from text.symbols import symbols
from utils.tools import get_mask_from_lengths


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradientReversal.apply(x, lambd)


class PhonemeAdversarialClassifier(nn.Module):
    def __init__(self, bottleneck_dim, num_classes, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, stylecode, grl_lambda=1.0):
        return self.net(grad_reverse(stylecode, grl_lambda))


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, codebook_size, commitment_weight, codebook_weight):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, inputs, valid_mask=None):
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        codebook = self.codebook.weight
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, codebook.t())
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        flat_indices = torch.argmin(distances, dim=1)
        indices = flat_indices.view(inputs.shape[:-1])
        quantized = self.codebook(flat_indices).view_as(inputs)

        if valid_mask is None:
            valid_mask = torch.ones(indices.shape, dtype=torch.bool, device=inputs.device)
        valid_inputs = inputs[valid_mask]
        valid_quantized = quantized[valid_mask]
        zero = inputs.new_zeros(())

        if valid_inputs.numel() == 0:
            commitment_loss = zero
            codebook_loss = zero
            vq_loss = zero
            code_usage = inputs.new_zeros(self.codebook_size)
            codebook_perplexity = zero
            used_code_count = zero
        else:
            commitment_loss = F.mse_loss(valid_inputs, valid_quantized.detach())
            codebook_loss = F.mse_loss(valid_quantized, valid_inputs.detach())
            vq_loss = self.codebook_weight * codebook_loss + self.commitment_weight * commitment_loss
            valid_indices = indices[valid_mask]
            code_usage = torch.bincount(valid_indices, minlength=self.codebook_size).to(dtype=inputs.dtype)
            probabilities = code_usage / code_usage.sum().clamp_min(1.0)
            nonzero = probabilities > 0
            codebook_perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
            used_code_count = (code_usage > 0).sum().to(dtype=inputs.dtype)

        quantized = inputs + (quantized - inputs).detach()
        indices = indices.masked_fill(~valid_mask, -1)
        return quantized, {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "code_indices": indices,
            "code_usage": code_usage,
            "codebook_perplexity": codebook_perplexity,
            "used_code_count": used_code_count,
            "codebook_size": self.codebook_size,
        }


class PhonemeStyleExtractor(nn.Module):
    """Extract phoneme-level style from mel targets using duration-aligned pooling."""

    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.hidden_size = model_config["transformer"]["encoder_hidden"]
        mel_dim = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        style_config = model_config.get("phoneme_style", {})
        frame_hidden = style_config.get("frame_hidden", self.hidden_size)
        conv_layers = style_config.get("conv_layers", 2)
        kernel_size = style_config.get("conv_kernel_size", 5)
        dropout = style_config.get("dropout", 0.1)
        self.bottleneck_dim = style_config.get("bottleneck_dim", 32)
        self.strict_duration_check = style_config.get("strict_duration_check", True)

        adversarial_config = style_config.get("adversarial", {})
        self.adv_enabled = adversarial_config.get("enabled", False)
        self.adv_weight = adversarial_config.get("weight", 0.0)
        self.adv_grl_lambda = adversarial_config.get("grl_lambda", 1.0)

        vq_config = style_config.get("vq", {})
        self.vq_enabled = vq_config.get("enabled", False)
        self.vector_quantizer = None

        self.input_projection = nn.Linear(mel_dim, frame_hidden)
        conv_blocks = []
        for layer in range(conv_layers):
            conv_blocks.extend(
                [
                    (
                        "conv_{}".format(layer + 1),
                        Conv(
                            frame_hidden,
                            frame_hidden,
                            kernel_size=kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                    ),
                    ("relu_{}".format(layer + 1), nn.ReLU()),
                    ("layer_norm_{}".format(layer + 1), nn.LayerNorm(frame_hidden)),
                    ("dropout_{}".format(layer + 1), nn.Dropout(dropout)),
                ]
            )
        self.frame_encoder = nn.Sequential(OrderedDict(conv_blocks))
        self.to_bottleneck = nn.Sequential(
            nn.Linear(frame_hidden, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
        )
        self.to_hidden = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
        )
        self.phoneme_adversary = None
        if self.adv_enabled:
            self.phoneme_adversary = PhonemeAdversarialClassifier(
                self.bottleneck_dim,
                len(symbols) + 1,
                adversarial_config.get("hidden", self.bottleneck_dim * 2),
                adversarial_config.get("dropout", dropout),
            )
        if self.vq_enabled:
            self.vector_quantizer = VectorQuantizer(
                self.bottleneck_dim,
                vq_config.get("codebook_size", 128),
                vq_config.get("commitment_weight", 0.25),
                vq_config.get("codebook_weight", 1.0),
            )

    def encode_frames(self, mel_targets):
        frame_hidden = self.input_projection(mel_targets)
        return self.frame_encoder(frame_hidden)

    def duration_pool(self, frame_hidden, duration_targets, src_lens=None, mel_lens=None):
        batch_size, mel_max_len, hidden = frame_hidden.shape
        max_src_len = duration_targets.shape[1]
        output = []

        for b in range(batch_size):
            src_len = (
                int(src_lens[b].item()) if src_lens is not None else max_src_len
            )
            mel_len = (
                int(mel_lens[b].item()) if mel_lens is not None else mel_max_len
            )
            durations = duration_targets[b, :src_len].long()
            duration_sum = int(durations.sum().item())
            if self.strict_duration_check and duration_sum > mel_len:
                raise ValueError(
                    "duration sum exceeds mel length for sample {}: src_len={}, mel_len={}, duration_sum={}".format(
                        b, src_len, mel_len, duration_sum
                    )
                )

            mel_seq = frame_hidden[b, :mel_len]
            phoneme_vectors = []
            cursor = 0
            for n, dur in enumerate(durations.tolist()):
                if dur <= 0:
                    phoneme_vectors.append(torch.zeros(hidden, device=frame_hidden.device))
                    continue
                end = cursor + dur
                if self.strict_duration_check and end > mel_len:
                    raise ValueError(
                        "duration segment exceeds mel length for sample {} phoneme {}: mel_len={}, cursor={}, duration={}".format(
                            b, n, mel_len, cursor, dur
                        )
                    )
                segment = mel_seq[cursor : min(end, mel_len)]
                if segment.numel() == 0:
                    phoneme_vectors.append(torch.zeros(hidden, device=frame_hidden.device))
                else:
                    phoneme_vectors.append(segment.mean(dim=0))
                cursor = end

            if src_len < max_src_len:
                phoneme_vectors.extend(
                    [
                        torch.zeros(hidden, device=frame_hidden.device)
                        for _ in range(max_src_len - src_len)
                    ]
                )
            output.append(torch.stack(phoneme_vectors, dim=0))

        return torch.stack(output, dim=0)

    def _get_invalid_mask(self, max_len, src_lens=None, duration_targets=None):
        mask = None
        if src_lens is not None:
            mask = get_mask_from_lengths(src_lens, max_len)
        if duration_targets is not None:
            duration_mask = duration_targets[:, :max_len] <= 0
            mask = duration_mask if mask is None else mask | duration_mask
        return mask

    def _mask_invalid_positions(self, x, src_lens=None, duration_targets=None):
        mask = self._get_invalid_mask(x.shape[1], src_lens, duration_targets)
        if mask is None:
            return x
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

    def extract_continuous_stylecode(self, mel_targets, duration_targets, src_lens=None, mel_lens=None):
        frame_hidden = self.encode_frames(mel_targets)
        pooled = self.duration_pool(frame_hidden, duration_targets, src_lens, mel_lens)
        stylecode = self.to_bottleneck(pooled)
        return self._mask_invalid_positions(stylecode, src_lens, duration_targets)

    def quantize_stylecode(self, stylecode, src_lens=None, duration_targets=None):
        if self.vector_quantizer is None:
            return stylecode, None
        invalid_mask = self._get_invalid_mask(stylecode.shape[1], src_lens, duration_targets)
        valid_mask = None if invalid_mask is None else ~invalid_mask
        quantized, style_info = self.vector_quantizer(stylecode, valid_mask=valid_mask)
        quantized = self._mask_invalid_positions(quantized, src_lens, duration_targets)
        return quantized, style_info

    def extract_stylecode(self, mel_targets, duration_targets, src_lens=None, mel_lens=None):
        stylecode, _ = self.extract_stylecode_with_info(
            mel_targets,
            duration_targets,
            src_lens=src_lens,
            mel_lens=mel_lens,
        )
        return stylecode

    def extract_stylecode_with_info(
        self,
        mel_targets,
        duration_targets,
        src_lens=None,
        mel_lens=None,
        include_continuous=False,
    ):
        continuous_stylecode = self.extract_continuous_stylecode(
            mel_targets,
            duration_targets,
            src_lens=src_lens,
            mel_lens=mel_lens,
        )
        stylecode, style_info = self.quantize_stylecode(
            continuous_stylecode,
            src_lens=src_lens,
            duration_targets=duration_targets,
        )
        if style_info is not None and include_continuous:
            style_info = dict(style_info)
            style_info["continuous_stylecode"] = continuous_stylecode
        return stylecode, style_info

    def decode_stylecode(self, stylecode, src_lens=None, duration_targets=None):
        style_phoneme = self.to_hidden(stylecode)
        return self._mask_invalid_positions(style_phoneme, src_lens, duration_targets)

    def classify_stylecode(self, stylecode, grl_lambda=None):
        if not self.adv_enabled or self.phoneme_adversary is None:
            return None
        if grl_lambda is None:
            grl_lambda = self.adv_grl_lambda
        return self.phoneme_adversary(stylecode, grl_lambda)

    def forward(self, mel_targets, duration_targets, src_lens=None, mel_lens=None):
        if mel_targets is None or duration_targets is None:
            raise ValueError("mel_targets and duration_targets are required for style extraction.")
        stylecode = self.extract_stylecode(mel_targets, duration_targets, src_lens, mel_lens)
        return self.decode_stylecode(stylecode, src_lens, duration_targets)
