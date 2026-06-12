import torch
import torch.nn as nn

from .mel_flow import masked_flow_matching_loss
from utils.tools import get_mask_from_lengths


class FastSpeech2Loss(nn.Module):
    """Flow-matching mel loss with duration, adversarial, and optional VQ losses."""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        loss_config = model_config.get("loss", {})
        adversarial_config = model_config.get("phoneme_style", {}).get("adversarial", {})
        self.duration_weight = loss_config.get("duration_weight", 1.0)
        self.flow_mel_weight = loss_config.get("flow_mel_weight", 1.0)
        self.adv_enabled = adversarial_config.get("enabled", False)
        self.adv_weight = adversarial_config.get("weight", 0.0)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, predictions):
        duration_targets = inputs[9]
        (
            _,
            v_pred,
            v_target,
            log_duration_predictions,
            src_masks,
            mel_masks,
            _,
            _,
            _,
        ) = predictions[:9]
        style_adv_logits = predictions[9] if len(predictions) > 9 else None
        style_info = predictions[10] if len(predictions) > 10 else None

        valid_src_masks = ~src_masks
        log_duration_targets = torch.log1p(duration_targets.float())
        log_duration_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(valid_src_masks)
        log_duration_targets = log_duration_targets.masked_select(valid_src_masks)

        flow_mel_loss = masked_flow_matching_loss(v_pred, v_target, padding_mask=mel_masks)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        phoneme_adv_loss = flow_mel_loss.new_zeros(())
        if self.adv_enabled and self.adv_weight > 0 and style_adv_logits is not None:
            n = style_adv_logits.shape[1]
            adv_targets = inputs[3][:, :n]
            adv_durations = duration_targets[:, :n]
            adv_pad_masks = get_mask_from_lengths(inputs[4], n)
            valid_adv_masks = ~(adv_pad_masks | (adv_targets == 0) | (adv_durations <= 0))
            if valid_adv_masks.any():
                phoneme_adv_loss = self.ce_loss(
                    style_adv_logits[:, :n, :][valid_adv_masks],
                    adv_targets[valid_adv_masks],
                )

        vq_loss = flow_mel_loss.new_zeros(())
        vq_commitment_loss = flow_mel_loss.new_zeros(())
        vq_codebook_loss = flow_mel_loss.new_zeros(())
        vq_codebook_perplexity = flow_mel_loss.new_zeros(())
        vq_used_code_count = flow_mel_loss.new_zeros(())
        if style_info is not None:
            vq_loss = style_info.get("vq_loss", vq_loss)
            vq_commitment_loss = style_info.get("commitment_loss", vq_commitment_loss)
            vq_codebook_loss = style_info.get("codebook_loss", vq_codebook_loss)
            vq_codebook_perplexity = style_info.get("codebook_perplexity", vq_codebook_perplexity)
            vq_used_code_count = style_info.get("used_code_count", vq_used_code_count)

        total_loss = (
            self.flow_mel_weight * flow_mel_loss
            + self.duration_weight * duration_loss
            + self.adv_weight * phoneme_adv_loss
            + vq_loss
        )

        losses = (
            total_loss,
            flow_mel_loss,
            duration_loss,
            phoneme_adv_loss,
        )
        if style_info is None:
            return losses
        return losses + (
            vq_loss,
            vq_commitment_loss,
            vq_codebook_loss,
            vq_codebook_perplexity,
            vq_used_code_count,
        )
