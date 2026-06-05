import torch


def lengths_to_padding_mask(lengths, max_len=None):
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)
    lengths = lengths.long()
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def make_valid_mask(lengths, durations=None, mask_zero_duration=True, max_len=None):
    padding_mask = lengths_to_padding_mask(lengths, max_len=max_len)
    valid_mask = ~padding_mask
    if durations is not None and mask_zero_duration:
        valid_mask = valid_mask & (durations[:, : valid_mask.shape[1]] > 0)
    return valid_mask


def mask_sequence(x, padding_mask, value=0.0):
    if padding_mask is None:
        return x
    while padding_mask.dim() < x.dim():
        padding_mask = padding_mask.unsqueeze(-1)
    return x.masked_fill(padding_mask, value)
