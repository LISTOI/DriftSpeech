import os
import json

import torch
import torch.nn as nn

from transformer import Encoder, Decoder, PostNet
from .modules import DurationPredictor, LengthRegulator
from .style_extractor import PhonemeStyleExtractor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        self.style_extractor = PhonemeStyleExtractor(preprocess_config, model_config)
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

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
                model_config["transformer"]["encoder_hidden"],
            )

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
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        text_hidden = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            text_hidden = text_hidden + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        log_duration_predictions = self.duration_predictor(text_hidden, src_masks)

        style_adv_logits = None
        style_info = None
        if mels is not None and d_targets is not None:
            stylecode, style_info = self.style_extractor.extract_stylecode_with_info(
                mels,
                d_targets,
                src_lens=src_lens,
                mel_lens=mel_lens,
            )
            style_adv_logits = self.style_extractor.classify_stylecode(stylecode)
            style_phoneme = self.style_extractor.decode_stylecode(
                stylecode,
                src_lens=src_lens,
                duration_targets=d_targets,
            )
        else:
            style_phoneme = torch.zeros_like(text_hidden)
        fused_hidden = text_hidden + style_phoneme

        if d_targets is not None:
            durations_for_lr = d_targets
        else:
            durations_for_lr = torch.clamp(
                torch.round(torch.exp(log_duration_predictions) - 1) * d_control,
                min=0,
            ).long()
            durations_for_lr = durations_for_lr.masked_fill(src_masks, 0)
            zero_duration_rows = durations_for_lr.sum(dim=1) == 0
            if zero_duration_rows.any():
                first_valid_positions = (~src_masks[zero_duration_rows]).float().argmax(dim=1)
                durations_for_lr[
                    zero_duration_rows.nonzero(as_tuple=True)[0], first_valid_positions.long()
                ] = 1
            max_mel_len = None

        expanded_hidden, mel_lens = self.length_regulator(
            fused_hidden, durations_for_lr, max_mel_len
        )
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        decoder_output, mel_masks = self.decoder(expanded_hidden, mel_masks)
        mel_predictions = self.mel_linear(decoder_output)

        postnet_output = self.postnet(mel_predictions) + mel_predictions

        return (
            mel_predictions,
            postnet_output,
            log_duration_predictions,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            style_adv_logits,
            style_info,
        )
