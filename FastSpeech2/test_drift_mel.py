import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parent / "model" / "drift_mel.py"
spec = importlib.util.spec_from_file_location("drift_mel", MODULE_PATH)
drift_mel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift_mel)
MelDriftGenerator = drift_mel.MelDriftGenerator
compute_V = drift_mel.compute_V


def main():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pos = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    y_neg = x.clone()
    drift = compute_V(x, y_pos, y_neg, temperature=0.5, mask_self=True)
    assert drift.shape == x.shape
    assert torch.isfinite(drift).all().item()

    torch.manual_seed(1234)
    generator = MelDriftGenerator(
        mel_dim=4,
        hidden_dim=8,
        config={
            "depth": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "alpha_embed_dim": 8,
            "max_seq_len": 8,
            "temperatures": [0.5],
            "max_tokens_per_utterance": 4,
            "cfg": {"enabled": True, "condition_dropout": 0.0, "guidance_scale": 1.5},
            "anchor": {"enabled": True, "type": "l1", "weight": 0.1},
        },
    )
    mel_targets = torch.randn(2, 5, 4)
    frame_condition = torch.randn(2, 5, 8)
    padding_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, False, False, True],
        ]
    )
    mel_pred, mel_target, info = generator.training_step(
        mel_targets,
        frame_condition,
        padding_mask=padding_mask,
    )
    assert mel_pred.shape == mel_targets.shape
    assert mel_target.shape == mel_targets.shape
    assert info["backend"] == "drift"
    assert info["sample_steps"] == 1
    assert torch.isfinite(info["mel_loss"]).item()
    assert torch.isfinite(info["drift_loss"]).item()
    assert torch.isfinite(info["anchor_loss"]).item()
    assert mel_pred[padding_mask].abs().max().item() == 0.0

    sample = generator.euler_sample(
        frame_condition,
        padding_mask=padding_mask,
        steps=8,
        guidance_scale=1.5,
    )
    assert sample.shape == mel_targets.shape
    assert torch.isfinite(sample).all().item()
    assert sample[padding_mask].abs().max().item() == 0.0
    print("drift_mel smoke test ok")


if __name__ == "__main__":
    main()
