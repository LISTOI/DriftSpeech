import torch
import torch.nn as nn

from utils.tools import get_mask_from_lengths


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.duration_weight = model_config.get("loss", {}).get("duration_weight", 1.0)
        adversarial_config = model_config.get("phoneme_style", {}).get("adversarial", {})
        self.adv_enabled = adversarial_config.get("enabled", False)
        self.adv_weight = adversarial_config.get("weight", 0.0)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, predictions):
        mel_targets = inputs[6]
        duration_targets = inputs[9]
        style_adv_logits = predictions[7] if len(predictions) > 7 else None
        style_info = predictions[8] if len(predictions) > 8 else None
        (
            mel_predictions,
            postnet_mel_predictions,
            log_duration_predictions,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions[:7]
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]
        log_duration_targets = torch.log1p(duration_targets.float())
        mel_targets.requires_grad = False
        log_duration_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        phoneme_adv_loss = mel_loss.new_zeros(())
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
        vq_loss = mel_loss.new_zeros(())
        vq_commitment_loss = mel_loss.new_zeros(())
        vq_codebook_loss = mel_loss.new_zeros(())
        vq_codebook_perplexity = mel_loss.new_zeros(())
        vq_used_code_count = mel_loss.new_zeros(())
        if style_info is not None:
            vq_loss = style_info.get("vq_loss", vq_loss)
            vq_commitment_loss = style_info.get("commitment_loss", vq_commitment_loss)
            vq_codebook_loss = style_info.get("codebook_loss", vq_codebook_loss)
            vq_codebook_perplexity = style_info.get("codebook_perplexity", vq_codebook_perplexity)
            vq_used_code_count = style_info.get("used_code_count", vq_used_code_count)

        total_loss = (
            mel_loss
            + postnet_mel_loss
            + self.duration_weight * duration_loss
            + self.adv_weight * phoneme_adv_loss
            + vq_loss
        )

        losses = (
            total_loss,
            mel_loss,
            postnet_mel_loss,
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
