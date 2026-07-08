import os
import json

import torch
import torch.nn as nn

from transformer import Encoder
from .drift_mel import MelDriftGenerator
from .mel_flow import FrameConditionSmoother, MelFlowGenerator
from .modules import DurationPredictor, LengthRegulator
from .style_extractor import PhonemeStyleExtractor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """FastSpeech2 text encoder with stylecode-conditioned flow mel generation."""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.train_config = None
        transformer_config = model_config["transformer"]
        self.mel_backend = model_config.get("mel_backend", "flow")
        mel_flow_config = model_config.get("mel_flow", {})
        mel_drift_config = model_config.get("mel_drift", {})
        backend_config = mel_drift_config if self.mel_backend == "drift" else mel_flow_config
        hidden_dim = backend_config.get("hidden_dim", transformer_config["encoder_hidden"])
        if hidden_dim != transformer_config["encoder_hidden"]:
            raise ValueError("{}.hidden_dim must match transformer.encoder_hidden in this branch".format(
                "mel_drift" if self.mel_backend == "drift" else "mel_flow"
            ))

        self.encoder = Encoder(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        self.style_extractor = PhonemeStyleExtractor(preprocess_config, model_config)
        self.length_regulator = LengthRegulator()
        self.condition_smoother = FrameConditionSmoother(
            hidden_dim,
            layers=backend_config.get("conv_smoother_layers", mel_flow_config.get("conv_smoother_layers", 4)),
            kernel_size=backend_config.get("conv_smoother_kernel_size", mel_flow_config.get("conv_smoother_kernel_size", 7)),
            dropout=backend_config.get("dropout", mel_flow_config.get("dropout", 0.1)),
            expansion=backend_config.get("conv_smoother_expansion", mel_flow_config.get("conv_smoother_expansion", 4)),
        )
        mel_dim = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        if self.mel_backend == "flow":
            self.mel_flow = MelFlowGenerator(
                mel_dim,
                hidden_dim,
                mel_flow_config,
            )
        elif self.mel_backend == "drift":
            self.mel_flow = MelDriftGenerator(
                mel_dim,
                hidden_dim,
                mel_drift_config,
            )
        else:
            raise ValueError("Unsupported mel_backend: {}".format(self.mel_backend))

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                transformer_config["encoder_hidden"],
            )

    def set_train_config(self, train_config):
        self.train_config = train_config

    def _predict_durations(self, log_duration_predictions, src_masks, d_control):
        durations = torch.clamp(
            torch.round(torch.exp(log_duration_predictions) - 1) * d_control,
            min=0,
        ).long()
        durations = durations.masked_fill(src_masks, 0)
        zero_duration_rows = durations.sum(dim=1) == 0
        if zero_duration_rows.any():
            first_valid_positions = (~src_masks[zero_duration_rows]).float().argmax(dim=1)
            durations[
                zero_duration_rows.nonzero(as_tuple=True)[0], first_valid_positions.long()
            ] = 1
        return durations

    def _build_style_hidden(self, mels, d_targets, src_lens, mel_lens, text_hidden, stylecode_override=None):
        style_adv_logits = None
        style_info = None
        if stylecode_override is not None:
            if d_targets is None:
                raise ValueError("d_targets are required when stylecode_override is provided")
            style_hidden = self.style_extractor.decode_stylecode(
                stylecode_override,
                src_lens=src_lens,
                duration_targets=d_targets,
            )
            style_info = {"source": "stylecode_override"}
            return style_hidden, style_adv_logits, style_info
        if mels is None or d_targets is None:
            return torch.zeros_like(text_hidden), style_adv_logits, style_info

        stylecode, style_info = self.style_extractor.extract_stylecode_with_info(
            mels,
            d_targets,
            src_lens=src_lens,
            mel_lens=mel_lens,
        )
        style_adv_logits = self.style_extractor.classify_stylecode(stylecode)
        style_hidden = self.style_extractor.decode_stylecode(
            stylecode,
            src_lens=src_lens,
            duration_targets=d_targets,
        )
        return style_hidden, style_adv_logits, style_info

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        d_targets=None,
        d_control=1.0,
        force_sampling=False,
        stylecode_override=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        text_hidden = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            text_hidden = text_hidden + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        log_duration_predictions = self.duration_predictor(text_hidden, src_masks)
        style_hidden, style_adv_logits, style_info = self._build_style_hidden(
            mels,
            d_targets,
            src_lens,
            mel_lens,
            text_hidden,
            stylecode_override=stylecode_override,
        )
        phoneme_condition = text_hidden + style_hidden

        if d_targets is not None:
            durations_for_lr = d_targets
        else:
            durations_for_lr = self._predict_durations(
                log_duration_predictions,
                src_masks,
                d_control,
            )
            max_mel_len = None

        frame_hidden, mel_lens = self.length_regulator(
            phoneme_condition,
            durations_for_lr,
            max_mel_len,
        )
        mel_masks = get_mask_from_lengths(mel_lens, frame_hidden.shape[1])
        frame_condition = self.condition_smoother(frame_hidden, mel_masks)

        if mels is not None and not force_sampling:
            mel_targets = mels[:, : frame_condition.shape[1], :]
            v_pred, v_target, flow_info = self.mel_flow.training_step(
                mel_targets,
                frame_condition,
                padding_mask=mel_masks,
            )
            if self.mel_backend == "drift":
                mel_predictions = v_pred
            else:
                mel_predictions = mel_targets + v_pred - v_target
            mel_predictions = mel_predictions.masked_fill(mel_masks.unsqueeze(-1), 0.0)
        else:
            sample_steps = 32
            guidance_scale = None
            if self.train_config is not None:
                sample_config = self.train_config.get("mel_drift", {}) if self.mel_backend == "drift" else {}
                flow_sample_config = self.train_config.get("mel_flow", {})
                sample_steps = sample_config.get("sample_steps", flow_sample_config.get("sample_steps", sample_steps))
                guidance_scale = sample_config.get("guidance_scale", flow_sample_config.get("guidance_scale", guidance_scale))
            mel_predictions = self.mel_flow.euler_sample(
                frame_condition,
                padding_mask=mel_masks,
                steps=sample_steps,
                guidance_scale=guidance_scale,
            )
            v_pred = mel_predictions
            v_target = torch.zeros_like(v_pred)
            flow_info = {
                "backend": self.mel_backend,
                "sample_steps": 1 if self.mel_backend == "drift" else sample_steps,
                "requested_sample_steps": sample_steps,
                "guidance_scale": guidance_scale,
            }

        return (
            mel_predictions,
            v_pred,
            v_target,
            log_duration_predictions,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            flow_info,
            style_adv_logits,
            style_info,
        )
