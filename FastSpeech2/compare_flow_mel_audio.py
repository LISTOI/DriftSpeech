import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path


AUDIO_ORDER = [
    ("00_GT", "GT"),
    ("01_baseline_nostylecode", "baseline_nostylecode"),
    ("02_method1_GTstylecode", "method1_GTstylecode"),
    ("03_method2_predstylecode", "method2_predstylecode"),
]


def sample_id_from_baseline_name(filename):
    stem = Path(filename).stem
    if stem.endswith("_gt"):
        return None
    if stem.endswith("_generated"):
        return stem[: -len("_generated")]
    return stem


def audio_tags(sample_id):
    return ["{}/{}".format(sample_id, name) for name, _ in AUDIO_ORDER]


def select_ids(metadata_ids, baseline_ids, max_samples):
    selected = [sample_id for sample_id in sorted(baseline_ids) if sample_id in set(metadata_ids)]
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_baseline_wavs(baseline_dir):
    baseline_dir = Path(baseline_dir)
    result = {}
    for path in sorted(baseline_dir.rglob("*.wav")):
        sample_id = sample_id_from_baseline_name(path.name)
        if sample_id is None:
            continue
        current = result.get(sample_id)
        if current is None or path.stem.endswith("_generated"):
            result[sample_id] = str(path)
    return result


def load_configs(args):
    import yaml

    with open(args.preprocess_config, "r", encoding="utf-8") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def build_selected_loader(metadata, sample_ids, preprocess_config, train_config, batch_size):
    from torch.utils.data import DataLoader

    from dataset import Dataset

    export_config = deepcopy(train_config)
    export_config["optimizer"] = dict(export_config["optimizer"])
    export_config["optimizer"]["batch_size"] = batch_size
    dataset = Dataset(metadata, preprocess_config, export_config, sort=False, drop_last=False)
    by_id = {sample_id: index for index, sample_id in enumerate(dataset.basename)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise RuntimeError("Baseline sample ids missing from {}: {}".format(metadata, ",".join(missing[:10])))
    indices = [by_id[sample_id] for sample_id in sample_ids]
    dataset.basename = [dataset.basename[index] for index in indices]
    dataset.speaker = [dataset.speaker[index] for index in indices]
    dataset.text = [dataset.text[index] for index in indices]
    dataset.raw_text = [dataset.raw_text[index] for index in indices]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def load_audio_for_tensorboard(path):
    from scipy.io import wavfile
    import numpy as np

    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max(scale, 1.0)
    else:
        audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return sample_rate, audio


def write_audio(writer, tag, audio, step, sample_rate):
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    writer.add_audio(tag, audio, global_step=step, sample_rate=sample_rate)


def save_and_log_audio(writer, path, tag, audio, step, sample_rate):
    from scipy.io import wavfile
    import numpy as np

    ensure_dir(os.path.dirname(path))
    clipped = np.clip(audio, -32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, clipped)
    write_audio(writer, tag, clipped, step, sample_rate)


def vocode_mels(mels, mel_lens, vocoder, model_config, preprocess_config, vocoder_infer):
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    lengths = mel_lens * hop_length
    return vocoder_infer(
        mels.transpose(1, 2),
        vocoder,
        model_config,
        preprocess_config,
        lengths=lengths,
    )


def synthesize_comparison_batch(
    batch,
    model,
    style_predictor,
    style_predictor_steps,
    style_predictor_seed,
    vocoder,
    configs,
    vocoder_infer,
    batch_index,
):
    import torch

    from evaluate_flow_mel import sample_with_gt_duration, style_predictor_batch_seed

    preprocess_config, model_config, _ = configs
    with torch.inference_mode():
        method1_output = sample_with_gt_duration(model, batch)
        method2_output = sample_with_gt_duration(
            model,
            batch,
            style_predictor=style_predictor,
            style_predictor_steps=style_predictor_steps,
            style_predictor_seed=style_predictor_batch_seed(style_predictor_seed, batch_index),
        )
        gt_audio = vocode_mels(batch[6], batch[7], vocoder, model_config, preprocess_config, vocoder_infer)
        method1_audio = vocode_mels(method1_output[0], method1_output[7], vocoder, model_config, preprocess_config, vocoder_infer)
        method2_audio = vocode_mels(method2_output[0], method2_output[7], vocoder, model_config, preprocess_config, vocoder_infer)
    return gt_audio, method1_audio, method2_audio


def run(args):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from evaluate_flow_mel import load_style_predictor, set_seed
    from utils.model import get_model, get_vocoder, move_vocoder, vocoder_infer
    from utils.tools import to_device

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    wav_root = os.path.join(args.output_dir, "wavs")
    log_dir = args.log_dir or os.path.join(args.output_dir, "tensorboard")
    ensure_dir(wav_root)
    ensure_dir(log_dir)

    baseline_wavs = find_baseline_wavs(args.baseline_dir)
    if not baseline_wavs:
        raise RuntimeError("No baseline wavs found in {}".format(args.baseline_dir))

    preprocess_config, model_config, train_config = load_configs(args)
    metadata_ids = []
    metadata_path = os.path.join(preprocess_config["path"]["preprocessed_path"], args.metadata)
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata_ids.append(line.split("|", 1)[0])
    selected_ids = select_ids(metadata_ids, set(baseline_wavs), args.max_samples)
    if not selected_ids:
        raise RuntimeError("No baseline wav ids matched {}".format(args.metadata))

    _, loader = build_selected_loader(args.metadata, selected_ids, preprocess_config, train_config, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_args = argparse.Namespace(restore_step=args.restore_step)
    train_config = deepcopy(train_config)
    for key in ("mel_drift", "mel_flow"):
        train_config.setdefault(key, {})["sample_steps"] = args.sample_steps
        train_config.setdefault(key, {})["guidance_scale"] = args.guidance_scale
    configs = (preprocess_config, model_config, train_config)
    model = get_model(model_args, configs, device, train=False).to(device)
    if hasattr(model, "set_train_config"):
        model.set_train_config(train_config)
    vocoder = get_vocoder(model_config, device)
    move_vocoder(vocoder, model_config, device)
    style_predictor = load_style_predictor(args.style_predictor_checkpoint, args.style_predictor_dir, device)
    if style_predictor is None:
        raise RuntimeError("--style_predictor_checkpoint is required for method2 predicted stylecode")

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    writer = SummaryWriter(log_dir)
    manifest = {
        "baseline_dir": args.baseline_dir,
        "output_dir": args.output_dir,
        "log_dir": log_dir,
        "metadata": args.metadata,
        "restore_step": args.restore_step,
        "style_predictor_checkpoint": args.style_predictor_checkpoint,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "sample_ids": selected_ids,
        "samples": [],
    }

    sample_step = 0
    batch_index = 0
    try:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                gt_audio, method1_audio, method2_audio = synthesize_comparison_batch(
                    batch,
                    model,
                    style_predictor,
                    args.style_predictor_steps,
                    args.style_predictor_seed,
                    vocoder,
                    configs,
                    vocoder_infer,
                    batch_index,
                )
                batch_index += 1
                for i, sample_id in enumerate(batch[0]):
                    sample_dir = os.path.join(wav_root, sample_id)
                    ensure_dir(sample_dir)
                    tags = audio_tags(sample_id)

                    gt_path = os.path.join(sample_dir, "00_GT.wav")
                    baseline_path = os.path.join(sample_dir, "01_baseline_nostylecode.wav")
                    method1_path = os.path.join(sample_dir, "02_method1_GTstylecode.wav")
                    method2_path = os.path.join(sample_dir, "03_method2_predstylecode.wav")

                    save_and_log_audio(writer, gt_path, tags[0], gt_audio[i], sample_step, sampling_rate)
                    shutil.copy2(baseline_wavs[sample_id], baseline_path)
                    baseline_rate, baseline_audio = load_audio_for_tensorboard(baseline_path)
                    write_audio(writer, tags[1], baseline_audio, sample_step, baseline_rate)
                    save_and_log_audio(writer, method1_path, tags[2], method1_audio[i], sample_step, sampling_rate)
                    save_and_log_audio(writer, method2_path, tags[3], method2_audio[i], sample_step, sampling_rate)

                    manifest["samples"].append(
                        {
                            "sample_id": sample_id,
                            "tensorboard_step": sample_step,
                            "tags": tags,
                            "gt_wav": gt_path,
                            "baseline_wav": baseline_path,
                            "method1_wav": method1_path,
                            "method2_wav": method2_path,
                            "baseline_source": baseline_wavs[sample_id],
                        }
                    )
                    sample_step += 1
    finally:
        writer.close()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    write_json(manifest_path, manifest)
    print("Wrote {} samples".format(len(manifest["samples"])))
    print("TensorBoard log dir: {}".format(log_dir))
    print("Manifest: {}".format(manifest_path))
    return manifest


def self_test():
    assert sample_id_from_baseline_name("LJ005-0143_generated.wav") == "LJ005-0143"
    assert sample_id_from_baseline_name("LJ005-0143.wav") == "LJ005-0143"
    assert sample_id_from_baseline_name("LJ005-0143_gt.wav") is None
    assert audio_tags("LJ005-0143") == [
        "LJ005-0143/00_GT",
        "LJ005-0143/01_baseline_nostylecode",
        "LJ005-0143/02_method1_GTstylecode",
        "LJ005-0143/03_method2_predstylecode",
    ]
    assert select_ids(["b", "a", "c"], {"a", "c"}, 1) == ["a"]
    print("compare_flow_mel_audio self_test ok")


def parse_args():
    parser = argparse.ArgumentParser("Compare GT, baseline, GT-stylecode, and predicted-stylecode audio in TensorBoard")
    parser.add_argument("--self_test", action="store_true", help="Run helper self test and exit")
    parser.add_argument("--baseline_dir", type=str, default="", help="Directory containing existing baseline wavs, e.g. *_generated.wav")
    parser.add_argument("--restore_step", type=int, default=0, help="Flow-stylecode FastSpeech2 checkpoint step")
    parser.add_argument("--style_predictor_checkpoint", type=str, default="", help="StyleCodePredictor checkpoint for method2")
    parser.add_argument("--style_predictor_dir", type=str, default="", help="Path to StyleCodePredictor package root; defaults to ../StyleCodePredictor")
    parser.add_argument("--style_predictor_steps", type=int, default=0, help="Euler steps for StyleCodePredictor; 0 uses checkpoint config")
    parser.add_argument("--style_predictor_seed", type=int, default=1234)
    parser.add_argument("--metadata", type=str, default="val.txt", help="Metadata split containing baseline ids, e.g. val.txt or train.txt")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="", help="TensorBoard log dir; defaults to output_dir/tensorboard")
    parser.add_argument("-p", "--preprocess_config", type=str, default="")
    parser.add_argument("-m", "--model_config", type=str, default="")
    parser.add_argument("-t", "--train_config", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.self_test:
        self_test()
    else:
        missing = []
        if not parsed_args.baseline_dir:
            missing.append("--baseline_dir")
        if not parsed_args.restore_step:
            missing.append("--restore_step")
        if not parsed_args.style_predictor_checkpoint:
            missing.append("--style_predictor_checkpoint")
        if not parsed_args.output_dir:
            missing.append("--output_dir")
        if not parsed_args.preprocess_config:
            missing.append("--preprocess_config")
        if not parsed_args.model_config:
            missing.append("--model_config")
        if not parsed_args.train_config:
            missing.append("--train_config")
        if missing:
            raise SystemExit("Missing required arguments: {}".format(", ".join(missing)))
        run(parsed_args)
