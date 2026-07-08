import argparse
import csv
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DETAIL_FIELDS = [
    "restore_step",
    "guidance_scale",
    "sample_steps",
    "stylecode_source",
    "sample_id",
    "utmos_score",
    "wav_path",
]

SUMMARY_FIELDS = [
    "restore_step",
    "guidance_scale",
    "sample_steps",
    "stylecode_source",
    "seed",
    "num_samples",
    "utmos_mean",
    "utmos_std",
    "utmos_min",
    "utmos_max",
    "wav_dir",
]


def split_ints(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid integer list: {}".format(value)) from exc


def split_floats(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one float")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid float list: {}".format(value)) from exc


def split_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_positive_steps(values):
    bad = [value for value in values if value <= 0]
    if bad:
        raise argparse.ArgumentTypeError("sample_steps must be positive integers: {}".format(bad))
    return values


def positive_int_list(value):
    return validate_positive_steps(split_ints(value))


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected positive integer: {}".format(value)) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected positive integer: {}".format(value))
    return parsed


def format_number(value):
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def combo_name(restore_step, guidance_scale, sample_steps):
    return "step{}_cfg{}_steps{}".format(
        restore_step,
        format_number(guidance_scale),
        sample_steps,
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def score_summary(values):
    if not values:
        return {
            "utmos_mean": 0.0,
            "utmos_std": 0.0,
            "utmos_min": 0.0,
            "utmos_max": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "utmos_mean": float(array.mean()),
        "utmos_std": float(array.std()),
        "utmos_min": float(array.min()),
        "utmos_max": float(array.max()),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lengths_to_padding_mask(lengths, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.long().unsqueeze(1)


def denormalize_stylecode(stylecode, style_mean, style_std):
    if style_mean is None or style_std is None:
        return stylecode
    return stylecode * style_std.to(stylecode.device)[None, None, :] + style_mean.to(stylecode.device)[None, None, :]


def stylecode_source(style_predictor):
    return "predicted" if style_predictor is not None else "gt_reference"


def evaluation_protocol(style_predictor_checkpoint):
    if style_predictor_checkpoint:
        return "GT duration mel backend synthesis with predicted stylecode from StyleCodePredictor; duration predictor error is intentionally excluded."
    return "GT duration mel backend synthesis with GT/reference stylecode extracted from ground-truth mel; duration predictor error is intentionally excluded."


def style_predictor_batch_seed(base_seed, batch_index):
    if base_seed is None:
        return None
    return int(base_seed) + int(batch_index)


def add_style_predictor_to_path(style_predictor_dir):
    if not style_predictor_dir:
        style_predictor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "StyleCodePredictor"))
    if style_predictor_dir not in sys.path:
        sys.path.insert(0, style_predictor_dir)
    return style_predictor_dir


def load_style_predictor(checkpoint_path, style_predictor_dir, device):
    if not checkpoint_path:
        return None
    add_style_predictor_to_path(style_predictor_dir)
    from stylecode_predictor.checkpoint import load_checkpoint
    from stylecode_predictor.model import build_model_from_config

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = build_model_from_config(
        config,
        vocab_size=int(ckpt.get("vocab_size", config["model"]["vocab_size"])),
        style_dim=int(ckpt.get("style_dim", config["model"]["style_dim"])),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_(False)
    style_mean = ckpt.get("style_mean")
    style_std = ckpt.get("style_std")
    if style_mean is not None:
        style_mean = style_mean.to(device)
    if style_std is not None:
        style_std = style_std.to(device)
    return {
        "model": model,
        "config": config,
        "style_mean": style_mean,
        "style_std": style_std,
        "style_dim": int(ckpt.get("style_dim", config["model"]["style_dim"])),
        "checkpoint": checkpoint_path,
    }


def predict_stylecode(style_predictor, batch, num_steps, seed=None):
    if style_predictor is None:
        return None
    from stylecode_predictor.flow_matching import euler_sample

    config = style_predictor["config"]
    noise_scale = float(config.get("flow", {}).get("noise_scale", 1.0))
    if num_steps is None or num_steps <= 0:
        num_steps = int(config.get("inference", {}).get("num_steps", 32))
    tokens = batch[3]
    lengths = batch[4]
    durations = batch[9]
    padding_mask = lengths_to_padding_mask(lengths, max_len=tokens.shape[1])

    def sample():
        pred = euler_sample(
            style_predictor["model"],
            tokens,
            lengths,
            padding_mask,
            style_dim=style_predictor["style_dim"],
            durations=durations,
            num_steps=num_steps,
            noise_scale=noise_scale,
        )
        return denormalize_stylecode(pred, style_predictor["style_mean"], style_predictor["style_std"])

    if seed is None:
        return sample()
    cuda_devices = [tokens.device.index if tokens.device.index is not None else torch.cuda.current_device()] if tokens.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if tokens.is_cuda:
            torch.cuda.manual_seed_all(seed)
        return sample()


def load_configs(args):
    with open(args.preprocess_config, "r", encoding="utf-8") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def validate_runtime_args(args):
    missing = []
    if not args.restore_steps:
        missing.append("--restore_steps")
    if not args.output_dir:
        missing.append("--output_dir")
    if args.utmos_backend != "fake" and not args.utmos_model_dir:
        missing.append("--utmos_model_dir")
    if not args.preprocess_config:
        missing.append("--preprocess_config")
    if not args.model_config:
        missing.append("--model_config")
    if not args.train_config:
        missing.append("--train_config")
    if args.max_samples < 0:
        raise SystemExit("--max_samples must be >= 0")
    if missing:
        raise SystemExit("Missing required arguments: {}".format(", ".join(missing)))


def find_duplicates(values):
    seen = set()
    duplicates = []
    duplicate_seen = set()
    for value in values:
        if value in seen and value not in duplicate_seen:
            duplicates.append(value)
            duplicate_seen.add(value)
        seen.add(value)
    return duplicates


def batch_size_for_selection(configured_batch_size, sample_ids, max_samples):
    if sample_ids:
        selection_size = len(sample_ids)
    elif max_samples > 0:
        selection_size = max_samples
    else:
        selection_size = 16
    return max(1, min(configured_batch_size, selection_size))


def filter_dataset(dataset, sample_ids, max_samples):
    if sample_ids:
        duplicate_requested = find_duplicates(sample_ids)
        if duplicate_requested:
            raise ValueError("Duplicate sample_ids requested: {}".format(",".join(duplicate_requested)))

        indices_by_id = {}
        for idx, sample_id in enumerate(dataset.basename):
            indices_by_id.setdefault(sample_id, []).append(idx)
        duplicate_metadata = [
            sample_id for sample_id in sample_ids if len(indices_by_id.get(sample_id, [])) > 1
        ]
        if duplicate_metadata:
            raise ValueError(
                "Duplicate sample_ids in source metadata: {}".format(",".join(duplicate_metadata))
            )
        missing = [sample_id for sample_id in sample_ids if sample_id not in indices_by_id]
        if missing:
            raise ValueError("Missing sample_ids in source metadata: {}".format(",".join(missing)))
        indices = [indices_by_id[sample_id][0] for sample_id in sample_ids]
    elif max_samples > 0:
        indices = list(range(min(max_samples, len(dataset.basename))))
    else:
        indices = list(range(len(dataset.basename)))

    if not indices:
        raise ValueError("No samples selected from source metadata")

    selected_basenames = [dataset.basename[idx] for idx in indices]
    duplicate_selected = find_duplicates(selected_basenames)
    if duplicate_selected:
        raise ValueError(
            "Duplicate selected sample basenames in source metadata: {}".format(
                ",".join(duplicate_selected)
            )
        )

    dataset.basename = selected_basenames
    dataset.speaker = [dataset.speaker[idx] for idx in indices]
    dataset.text = [dataset.text[idx] for idx in indices]
    dataset.raw_text = [dataset.raw_text[idx] for idx in indices]
    return dataset


def build_loader(preprocess_config, train_config, source, sample_ids, max_samples):
    # If top-level imports would break self_test in torch_gpu, import Dataset/DataLoader lazily here.
    from dataset import Dataset
    from torch.utils.data import DataLoader

    eval_config = deepcopy(train_config)
    eval_config["optimizer"] = dict(eval_config["optimizer"])
    eval_config["optimizer"]["batch_size"] = batch_size_for_selection(
        eval_config["optimizer"].get("batch_size", 16),
        sample_ids,
        max_samples,
    )
    dataset = Dataset(source, preprocess_config, eval_config, sort=False, drop_last=False)
    dataset = filter_dataset(dataset, sample_ids, max_samples)
    loader = DataLoader(
        dataset,
        batch_size=eval_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


class UTMOSScorer:
    def __init__(self, model_dir, device, backend="torchhub"):
        self.backend = backend
        self.device = device
        self.model = None
        if backend == "fake":
            return
        if backend != "torchhub":
            raise ValueError("Unsupported UTMOS backend: {}".format(backend))
        if not model_dir:
            raise ValueError("--utmos_model_dir is required for torchhub backend")
        self.model = torch.hub.load(model_dir, "utmos22_strong", source="local")
        if hasattr(self.model, "to"):
            self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def score_wav(self, wav_path):
        if self.backend == "fake":
            return float((sum(bytearray(os.path.basename(wav_path), "utf-8")) % 300) / 100.0 + 2.0)
        sample_rate, audio = wavfile.read(wav_path)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        wav = torch.from_numpy(audio).float().to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.inference_mode():
            try:
                score = self.model(wav, sample_rate)
            except TypeError:
                try:
                    score = self.model(wav)
                except TypeError:
                    score = self.model(wav_path)
        if isinstance(score, (list, tuple)):
            score = score[0]
        if hasattr(score, "detach"):
            score = score.detach().float().cpu().reshape(-1)[0].item()
        return float(score)


def sample_with_gt_duration(
    model,
    batch,
    style_predictor=None,
    style_predictor_steps=None,
    style_predictor_seed=None,
    predict_stylecode_fn=predict_stylecode,
):
    stylecode_override = predict_stylecode_fn(
        style_predictor,
        batch,
        style_predictor_steps,
        style_predictor_seed,
    )
    use_gt_stylecode = stylecode_override is None
    return model(
        speakers=batch[2],
        texts=batch[3],
        src_lens=batch[4],
        max_src_len=batch[5],
        mels=batch[6] if use_gt_stylecode else None,
        mel_lens=batch[7] if use_gt_stylecode else None,
        max_mel_len=batch[8],
        d_targets=batch[9],
        force_sampling=True,
        stylecode_override=stylecode_override,
    )


def set_flow_sampling_config(model, train_config, guidance_scale, sample_steps):
    for key in ("mel_drift", "mel_flow"):
        train_config.setdefault(key, {})["guidance_scale"] = guidance_scale
        train_config.setdefault(key, {})["sample_steps"] = sample_steps
    if hasattr(model, "set_train_config"):
        model.set_train_config(train_config)


def vocode_mel(mel, vocoder, model_config, preprocess_config):
    # If vocoder_infer is not top-level imported, import lazily here.
    from utils.model import vocoder_infer

    return vocoder_infer(
        mel.transpose(0, 1).unsqueeze(0),
        vocoder,
        model_config,
        preprocess_config,
    )[0]


def write_audio(writer, tag, audio, global_step, sampling_rate):
    audio = np.asarray(audio, dtype=np.float32)
    scale = np.max(np.abs(audio)) if audio.size else 0.0
    if scale > 0:
        audio = audio / scale
    writer.add_audio(tag, audio, global_step=global_step, sample_rate=sampling_rate)


def select_preview_sample_ids(detail_rows, sample_count, seed):
    sample_ids = sorted({row["sample_id"] for row in detail_rows})
    if not sample_ids:
        return []
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    return sample_ids[: min(sample_count, len(sample_ids))]


def log_top_k_audio(writer, summary_rows, detail_rows, output_dir, top_k, sample_count, seed):
    if writer is None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output_dir)
        owns_writer = True
    else:
        owns_writer = False

    preview_sample_ids = select_preview_sample_ids(detail_rows, sample_count, seed)
    try:
        for rank, row in enumerate(summary_rows[: min(top_k, len(summary_rows))], start=1):
            name = combo_name(row["restore_step"], row["guidance_scale"], row["sample_steps"])
            source = row.get("stylecode_source", "gt_reference")
            writer.add_scalar(
                "TopAudio/{}/utmos_mean/rank{}".format(source, rank),
                row["utmos_mean"],
                row["restore_step"],
            )
            for sample_id in preview_sample_ids:
                wav_path = os.path.join(
                    output_dir,
                    "wav",
                    source,
                    name,
                    "{}.wav".format(sample_id),
                )
                if not os.path.exists(wav_path):
                    raise FileNotFoundError("Missing top-k preview wav: {}".format(wav_path))
                sampling_rate, audio = wavfile.read(wav_path)
                write_audio(
                    writer,
                    "TopAudio/{}/rank{}/{}/{}".format(source, rank, name, sample_id),
                    audio,
                    row["restore_step"],
                    sampling_rate,
                )
    finally:
        if owns_writer:
            writer.flush()
            writer.close()


def evaluate_combination(
    model,
    vocoder,
    loader,
    configs,
    restore_step,
    guidance_scale,
    sample_steps,
    seed,
    output_dir,
    scorer,
    writer=None,
    style_predictor=None,
    style_predictor_steps=None,
    style_predictor_seed=None,
):
    preprocess_config, model_config, train_config = configs
    set_seed(seed)
    set_flow_sampling_config(model, train_config, guidance_scale, sample_steps)
    source = stylecode_source(style_predictor)
    name = combo_name(restore_step, guidance_scale, sample_steps)
    wav_dir = os.path.join(output_dir, "wav", source, name)
    ensure_dir(wav_dir)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    detail_rows = []
    scores = []

    model.eval()
    with torch.inference_mode():
        batch_index = 0
        for batchs in loader:
            for batch in batchs:
                # If to_device is not top-level imported, import lazily before use.
                from utils.tools import to_device

                batch = to_device(batch, device)
                output = sample_with_gt_duration(
                    model,
                    batch,
                    style_predictor=style_predictor,
                    style_predictor_steps=style_predictor_steps,
                    style_predictor_seed=style_predictor_batch_seed(style_predictor_seed, batch_index),
                )
                batch_index += 1
                mel_predictions = output[0]
                mel_lens = output[7]
                for i, sample_id in enumerate(batch[0]):
                    mel_len = int(mel_lens[i].item())
                    mel_prediction = mel_predictions[i, :mel_len]
                    audio = vocode_mel(mel_prediction, vocoder, model_config, preprocess_config)
                    wav_path = os.path.join(wav_dir, "{}.wav".format(sample_id))
                    wavfile.write(wav_path, sampling_rate, np.asarray(audio, dtype=np.int16))
                    utmos_score = scorer.score_wav(wav_path)
                    scores.append(utmos_score)
                    detail_rows.append(
                        {
                            "restore_step": restore_step,
                            "guidance_scale": guidance_scale,
                            "sample_steps": sample_steps,
                            "stylecode_source": source,
                            "sample_id": sample_id,
                            "utmos_score": utmos_score,
                            "wav_path": wav_path,
                        }
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "UTMOS/sample/{}/step{}/cfg_{}/steps_{}/{}".format(
                                source,
                                restore_step,
                                format_number(guidance_scale),
                                sample_steps,
                                sample_id,
                            ),
                            utmos_score,
                            restore_step,
                        )
                        write_audio(
                            writer,
                            "Audio/{}/step{}/cfg_{}/steps_{}/{}".format(
                                source,
                                restore_step,
                                format_number(guidance_scale),
                                sample_steps,
                                sample_id,
                            ),
                            audio,
                            restore_step,
                            sampling_rate,
                        )

    summary = {
        "restore_step": restore_step,
        "guidance_scale": guidance_scale,
        "sample_steps": sample_steps,
        "stylecode_source": source,
        "seed": seed,
        "num_samples": len(scores),
        "wav_dir": wav_dir,
    }
    summary.update(score_summary(scores))
    if writer is not None:
        writer.add_scalar(
            "UTMOS/mean/{}/step{}/cfg_{}/steps_{}".format(
                source,
                restore_step,
                format_number(guidance_scale),
                sample_steps,
            ),
            summary["utmos_mean"],
            restore_step,
        )
    return summary, detail_rows


def self_test():
    assert split_ints("700000,800000,900000") == [700000, 800000, 900000]
    assert split_floats("1.0,1.5,2.0") == [1.0, 1.5, 2.0]
    assert split_strings("LJ001-0001,LJ002-0001") == ["LJ001-0001", "LJ002-0001"]
    assert split_strings("") == []
    assert format_number(1.5) == "1p5"
    assert combo_name(900000, 1.5, 16) == "step900000_cfg1p5_steps16"
    stats = score_summary([4.0, 3.0, 5.0])
    assert stats["utmos_mean"] == 4.0
    assert round(stats["utmos_std"], 6) == round(0.816496580927726, 6)
    assert stats["utmos_min"] == 3.0
    assert stats["utmos_max"] == 5.0
    assert validate_positive_steps([1, 8, 16, 32]) == [1, 8, 16, 32]
    try:
        validate_positive_steps([1, 0])
    except argparse.ArgumentTypeError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero sample step should fail")
    args = parse_args([])
    assert args.source == "val.txt"
    assert args.max_samples == 0
    assert args.guidance_scales == [1.0, 1.5, 2.0]
    assert args.sample_steps == [1, 8, 16, 32]
    assert args.style_predictor_checkpoint == ""
    assert args.style_predictor_steps == 0
    assert args.style_predictor_seed == 1234

    class FakeDataset:
        basename = ["a", "b", "c"]
        speaker = ["spk", "spk", "spk"]
        text = ["ta", "tb", "tc"]
        raw_text = ["raw a", "raw b", "raw c"]

    assert find_duplicates(["a", "b", "a", "b", "b"]) == ["a", "b"]
    assert batch_size_for_selection(16, ["a", "b", "c"], 1) == 3
    assert batch_size_for_selection(2, ["a", "b", "c"], 8) == 2
    assert batch_size_for_selection(16, [], 3) == 3
    assert batch_size_for_selection(8, [], 0) == 8

    filtered = filter_dataset(FakeDataset(), sample_ids=["c", "a"], max_samples=0)
    assert filtered.basename == ["c", "a"]
    assert filtered.raw_text == ["raw c", "raw a"]
    filtered = filter_dataset(FakeDataset(), sample_ids=["c", "a"], max_samples=1)
    assert filtered.basename == ["c", "a"]
    filtered = filter_dataset(FakeDataset(), sample_ids=[], max_samples=2)
    assert filtered.basename == ["a", "b"]
    try:
        filter_dataset(FakeDataset(), sample_ids=["missing"], max_samples=0)
    except ValueError as exc:
        assert "Missing sample_ids" in str(exc)
    else:
        raise AssertionError("missing sample id should fail")
    try:
        filter_dataset(FakeDataset(), sample_ids=["a", "a"], max_samples=0)
    except ValueError as exc:
        assert "Duplicate sample_ids requested" in str(exc)
    else:
        raise AssertionError("duplicate requested sample ids should fail")

    class DuplicateFakeDataset:
        basename = ["a", "b", "a"]
        speaker = ["spk", "spk", "spk"]
        text = ["ta", "tb", "tc"]
        raw_text = ["raw a1", "raw b", "raw a2"]

    filtered = filter_dataset(DuplicateFakeDataset(), sample_ids=[], max_samples=2)
    assert filtered.basename == ["a", "b"]
    try:
        filter_dataset(DuplicateFakeDataset(), sample_ids=["a"], max_samples=0)
    except ValueError as exc:
        assert "Duplicate sample_ids in source metadata" in str(exc)
    else:
        raise AssertionError("duplicate metadata basename should fail")
    try:
        filter_dataset(DuplicateFakeDataset(), sample_ids=[], max_samples=3)
    except ValueError as exc:
        assert "Duplicate selected sample basenames" in str(exc)
    else:
        raise AssertionError("duplicate max_samples basename should fail")
    try:
        filter_dataset(DuplicateFakeDataset(), sample_ids=[], max_samples=0)
    except ValueError as exc:
        assert "Duplicate selected sample basenames" in str(exc)
    else:
        raise AssertionError("duplicate all-mode basename should fail")

    scorer = UTMOSScorer("", torch.device("cpu"), backend="fake")
    assert scorer.score_wav("abc.wav") == scorer.score_wav("abc.wav")
    assert isinstance(scorer.score_wav("abc.wav"), float)

    assert stylecode_source(None) == "gt_reference"
    assert stylecode_source({"model": object()}) == "predicted"
    assert "GT/reference stylecode" in evaluation_protocol(None)
    assert "predicted stylecode" in evaluation_protocol("predictor.pt")

    class ConfigRecorder:
        def __init__(self):
            self.train_config = None

        def set_train_config(self, train_config):
            self.train_config = train_config

    recorder = ConfigRecorder()
    train_config = {}
    set_flow_sampling_config(recorder, train_config, 1.5, 1)
    assert recorder.train_config is train_config
    assert train_config["mel_drift"]["guidance_scale"] == 1.5
    assert train_config["mel_drift"]["sample_steps"] == 1
    assert train_config["mel_flow"]["guidance_scale"] == 1.5
    assert train_config["mel_flow"]["sample_steps"] == 1
    assert style_predictor_batch_seed(1234, 2) == 1236
    assert style_predictor_batch_seed(None, 2) is None
    stylecode = torch.ones(1, 2, 3)
    style_mean = torch.tensor([1.0, 2.0, 3.0])
    style_std = torch.tensor([2.0, 2.0, 2.0])
    denormalized = denormalize_stylecode(stylecode, style_mean, style_std)
    assert denormalized[0, 0].tolist() == [3.0, 4.0, 5.0]
    mask = lengths_to_padding_mask(torch.tensor([2, 3]), 4)
    assert mask.tolist() == [[False, False, True, True], [False, False, False, True]]

    class RecordingModel:
        def __call__(self, **kwargs):
            return kwargs

    fake_batch = (
        ["sample-a"],
        ["raw text"],
        torch.tensor([0]),
        torch.tensor([[1, 2, 0]]),
        torch.tensor([2]),
        3,
        torch.ones(1, 5, 80),
        torch.tensor([5]),
        5,
        torch.tensor([[2, 3, 0]]),
    )
    gt_call = sample_with_gt_duration(RecordingModel(), fake_batch)
    assert gt_call["mels"] is fake_batch[6]
    assert gt_call["mel_lens"] is fake_batch[7]
    assert gt_call["force_sampling"] is True
    assert gt_call["stylecode_override"] is None

    predicted_stylecode = torch.ones(1, 3, 32)
    predicted_call = sample_with_gt_duration(
        RecordingModel(),
        fake_batch,
        style_predictor={"predicted_stylecode": predicted_stylecode},
        style_predictor_steps=8,
        style_predictor_seed=1234,
        predict_stylecode_fn=lambda predictor, batch, steps, seed: predictor["predicted_stylecode"],
    )
    assert predicted_call["mels"] is None
    assert predicted_call["mel_lens"] is None
    assert predicted_call["force_sampling"] is True
    assert predicted_call["stylecode_override"] is predicted_stylecode

    assert positive_int("3") == 3
    try:
        positive_int("0")
    except argparse.ArgumentTypeError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero top-k value should fail")

    details_for_preview = [
        {"sample_id": "b"},
        {"sample_id": "a"},
        {"sample_id": "c"},
        {"sample_id": "a"},
    ]
    selected_once = select_preview_sample_ids(details_for_preview, 2, 1234)
    selected_twice = select_preview_sample_ids(details_for_preview, 2, 1234)
    assert selected_once == selected_twice
    assert len(selected_once) == 2
    assert set(selected_once).issubset({"a", "b", "c"})
    all_selected = select_preview_sample_ids(details_for_preview, 10, 1234)
    assert all_selected and len(all_selected) == 3

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_for_preview = [
            {
                "restore_step": 900000,
                "guidance_scale": 1.5,
                "sample_steps": 16,
                "utmos_mean": 4.1,
            },
            {
                "restore_step": 900000,
                "guidance_scale": 1.0,
                "sample_steps": 32,
                "utmos_mean": 4.0,
            },
        ]
        for row in summary_for_preview:
            row["stylecode_source"] = "gt_reference"
            name = combo_name(row["restore_step"], row["guidance_scale"], row["sample_steps"])
            wav_dir = os.path.join(tmpdir, "wav", row["stylecode_source"], name)
            ensure_dir(wav_dir)
            for sample_id in {"a", "b", "c"}:
                wavfile.write(
                    os.path.join(wav_dir, "{}.wav".format(sample_id)),
                    22050,
                    np.asarray([0, 1000, -1000], dtype=np.int16),
                )
        log_top_k_audio(None, summary_for_preview, details_for_preview, tmpdir, 2, 2, 1234)
        event_files = [name for name in os.listdir(tmpdir) if name.startswith("events.out.tfevents")]
        assert event_files
    print("evaluate_flow_mel_utmos self_test ok")


def run(args):
    ensure_dir(args.output_dir)
    base_configs = load_configs(args)
    preprocess_config, model_config, base_train_config = base_configs
    dataset, loader = build_loader(
        preprocess_config,
        base_train_config,
        args.source,
        args.sample_ids,
        args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("Selected dataset is empty")

    from utils.model import get_model, get_vocoder, move_vocoder

    scorer = UTMOSScorer(args.utmos_model_dir, device, backend=args.utmos_backend)
    vocoder = get_vocoder(model_config, device)
    style_predictor = load_style_predictor(args.style_predictor_checkpoint, args.style_predictor_dir, device)
    if style_predictor is not None:
        print("Loaded StyleCodePredictor checkpoint: {}".format(args.style_predictor_checkpoint))
    writer = None
    if args.log_tensorboard or args.log_top_k_audio:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.output_dir)
    summary_rows = []
    detail_rows = []

    try:
        for restore_step in args.restore_steps:
            model_args = argparse.Namespace(restore_step=restore_step)
            train_config = deepcopy(base_train_config)
            configs = (preprocess_config, model_config, train_config)
            model = get_model(model_args, configs, device, train=False).to(device)
            move_vocoder(vocoder, model_config, device)
            try:
                for guidance_scale in args.guidance_scales:
                    for sample_steps in args.sample_steps:
                        summary, details = evaluate_combination(
                            model,
                            vocoder,
                            loader,
                            configs,
                            restore_step,
                            guidance_scale,
                            sample_steps,
                            args.seed,
                            args.output_dir,
                            scorer,
                            writer=writer,
                            style_predictor=style_predictor,
                            style_predictor_steps=args.style_predictor_steps,
                            style_predictor_seed=args.style_predictor_seed,
                        )
                        summary_rows.append(summary)
                        detail_rows.extend(details)
                        print(
                            "step={}, cfg={}, steps={}, stylecode={}, samples={}, utmos_mean={:.6f}, utmos_std={:.6f}".format(
                                restore_step,
                                guidance_scale,
                                sample_steps,
                                summary["stylecode_source"],
                                summary["num_samples"],
                                summary["utmos_mean"],
                                summary["utmos_std"],
                            )
                        )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        move_vocoder(vocoder, model_config, torch.device("cpu"))

    summary_rows = sorted(summary_rows, key=lambda row: row["utmos_mean"], reverse=True)
    best = summary_rows[0]
    try:
        if args.log_top_k_audio:
            log_top_k_audio(
                writer,
                summary_rows,
                detail_rows,
                args.output_dir,
                args.top_k_audio,
                args.top_k_audio_samples,
                args.top_k_audio_seed,
            )
        write_csv(os.path.join(args.output_dir, "details.csv"), DETAIL_FIELDS, detail_rows)
        write_csv(os.path.join(args.output_dir, "summary.csv"), SUMMARY_FIELDS, summary_rows)
        write_json(os.path.join(args.output_dir, "summary.json"), {"rows": summary_rows})
        write_json(
            os.path.join(args.output_dir, "best.json"),
            {
                "best_by_utmos_mean": best,
                "score_rule": "higher utmos_mean is better",
                "ranking_note": "UTMOS is an automatic MOS predictor; confirm the final choice by listening to top candidates.",
                "evaluation_protocol": evaluation_protocol(args.style_predictor_checkpoint),
                "style_predictor_checkpoint": args.style_predictor_checkpoint,
                "style_predictor_steps": args.style_predictor_steps,
                "style_predictor_seed": args.style_predictor_seed,
            },
        )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    print("Best automatic combination by UTMOS mean:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return {"rows": summary_rows, "details": detail_rows, "best": best}


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate flow-mel checkpoints with offline UTMOS")
    parser.add_argument("--self_test", action="store_true", help="Run helper-function self test and exit")
    parser.add_argument("--restore_steps", type=split_ints, default=[])
    parser.add_argument("--guidance_scales", type=split_floats, default=split_floats("1.0,1.5,2.0"))
    parser.add_argument("--sample_steps", type=positive_int_list, default=positive_int_list("1,8,16,32"))
    parser.add_argument("--source", type=str, default="val.txt")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all selected samples")
    parser.add_argument("--sample_ids", type=split_strings, default=[])
    parser.add_argument("--utmos_model_dir", type=str, default="")
    parser.add_argument("--utmos_backend", type=str, default="torchhub", choices=["torchhub", "fake"])
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--log_tensorboard", action="store_true")
    parser.add_argument("--log_top_k_audio", action="store_true")
    parser.add_argument("--top_k_audio", type=positive_int, default=3)
    parser.add_argument("--top_k_audio_samples", type=positive_int, default=3)
    parser.add_argument("--top_k_audio_seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--style_predictor_checkpoint", type=str, default="", help="Optional StyleCodePredictor checkpoint; if set, predicted stylecode replaces GT/reference stylecode")
    parser.add_argument("--style_predictor_dir", type=str, default="", help="Path to StyleCodePredictor package root; defaults to ../StyleCodePredictor")
    parser.add_argument("--style_predictor_steps", type=int, default=0, help="Euler steps for StyleCodePredictor; 0 uses checkpoint config")
    parser.add_argument("--style_predictor_seed", type=int, default=1234, help="Seed for StyleCodePredictor sampling")
    parser.add_argument("-p", "--preprocess_config", type=str, default="")
    parser.add_argument("-m", "--model_config", type=str, default="")
    parser.add_argument("-t", "--train_config", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.self_test:
        self_test()
    else:
        validate_runtime_args(parsed_args)
        run(parsed_args)
