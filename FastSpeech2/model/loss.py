import torch
import torch.nn as nn

from .mel_flow import masked_flow_matching_loss


class FastSpeech2Loss(nn.Module):
    """Flow-matching mel loss plus duration loss."""

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        loss_config = model_config.get("loss", {})
        self.duration_weight = loss_config.get("duration_weight", 1.0)
        self.flow_mel_weight = loss_config.get("flow_mel_weight", 1.0)
        self.mse_loss = nn.MSELoss()

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
        ) = predictions

        valid_src_masks = ~src_masks
        log_duration_targets = torch.log1p(duration_targets.float())
        log_duration_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(valid_src_masks)
        log_duration_targets = log_duration_targets.masked_select(valid_src_masks)

        flow_mel_loss = masked_flow_matching_loss(v_pred, v_target, padding_mask=mel_masks)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        total_loss = self.flow_mel_weight * flow_mel_loss + self.duration_weight * duration_loss

        return (
            total_loss,
            flow_mel_loss,
            duration_loss,
        )
