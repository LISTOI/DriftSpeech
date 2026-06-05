import torch

from .masks import make_valid_mask, mask_sequence
from .metrics import masked_mse


def sample_flow_batch(stylecode, t_min=0.0, t_max=1.0, noise_scale=1.0):
    batch_size = stylecode.shape[0]
    x0 = torch.randn_like(stylecode) * noise_scale
    t = torch.empty(batch_size, device=stylecode.device, dtype=stylecode.dtype).uniform_(t_min, t_max)
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * stylecode
    v_target = stylecode - x0
    return x_t, t, v_target, x0


def flow_matching_loss(
    model,
    tokens,
    stylecode,
    lengths,
    padding_mask,
    durations=None,
    mask_zero_duration=True,
    t_min=0.0,
    t_max=1.0,
    noise_scale=1.0,
):
    x_t, t, v_target, _ = sample_flow_batch(
        stylecode,
        t_min=t_min,
        t_max=t_max,
        noise_scale=noise_scale,
    )
    x_t = mask_sequence(x_t, padding_mask)
    v_target = mask_sequence(v_target, padding_mask)
    v_pred = model(tokens, x_t, t, padding_mask, durations=durations)
    valid_mask = make_valid_mask(
        lengths,
        durations=durations,
        mask_zero_duration=mask_zero_duration,
        max_len=stylecode.shape[1],
    )
    return masked_mse(v_pred, v_target, valid_mask), {
        "v_pred": v_pred,
        "v_target": v_target,
        "x_t": x_t,
        "t": t,
        "valid_mask": valid_mask,
    }


@torch.no_grad()
def euler_sample(
    model,
    tokens,
    lengths,
    padding_mask,
    style_dim,
    durations=None,
    num_steps=32,
    noise_scale=1.0,
):
    z = torch.randn(tokens.shape[0], tokens.shape[1], style_dim, device=tokens.device) * noise_scale
    z = mask_sequence(z, padding_mask)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_value = torch.full((tokens.shape[0],), step / num_steps, device=tokens.device, dtype=z.dtype)
        v = model(tokens, z, t_value, padding_mask, durations=durations)
        z = z + dt * v
        z = mask_sequence(z, padding_mask)
    return z
