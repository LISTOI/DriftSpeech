import argparse
import csv
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch.utils.data import DataLoader

from dataset import Dataset
from utils.model import get_model, get_vocoder, move_vocoder, vocoder_infer
from utils.tools import plot_mel, to_device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def format_number(value):
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_summary(values):
    if not values:
        return {"mean": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def compute_mel_metrics(prediction, target, length):
    prediction = prediction[:length].detach().float().cpu().numpy()
    target = target[:length].detach().float().cpu().numpy()
    diff = prediction - target
    mse = float(np.mean(diff ** 2))
    return {
        "mel_l1": float(np.mean(np.abs(diff))),
        "mel_mse": mse,
        "mel_rmse": float(np.sqrt(mse)),
        "mel_mean_abs_diff": float(np.mean(np.abs(prediction.mean(axis=0) - target.mean(axis=0)))),
        "mel_std_abs_diff": float(np.mean(np.abs(prediction.std(axis=0) - target.std(axis=0)))),
    }


def score_row(row):
    return row["mel_l1_mean"] + 0.1 * row["mel_std_abs_diff_mean"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path, rows):
    fieldnames = [
        "restore_step",
        "guidance_scale",
        "sample_steps",
        "seed",
        "num_samples",
        "mel_l1_mean",
        "mel_l1_std",
        "mel_mse_mean",
        "mel_mse_std",
        "mel_rmse_mean",
        "mel_rmse_std",
        "mel_mean_abs_diff_mean",
        "mel_mean_abs_diff_std",
        "mel_std_abs_diff_mean",
        "mel_std_abs_diff_std",
        "score",
        "sample_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_mel_png(path, mel, title):
    fig = plot_mel([mel.detach().transpose(0, 1).cpu().numpy()], [title])
    fig.savefig(path)
    plt.close(fig)


def save_sample_artifacts(
    sample_dir,
    sample_id,
    mel_prediction,
    mel_target,
    mel_len,
    vocoder,
    model_config,
    preprocess_config,
):
    ensure_dir(sample_dir)
    mel_prediction = mel_prediction[:mel_len]
    mel_target = mel_target[:mel_len]
    save_mel_png(os.path.join(sample_dir, "{}_generated.png".format(sample_id)), mel_prediction, "Generated Mel")
    save_mel_png(os.path.join(sample_dir, "{}_gt.png".format(sample_id)), mel_target, "Ground-Truth Mel")

    if vocoder is None:
        return

    prediction_audio = vocoder_infer(
        mel_prediction.transpose(0, 1).unsqueeze(0),
        vocoder,
        model_config,
        preprocess_config,
    )[0]
    target_audio = vocoder_infer(
        mel_target.transpose(0, 1).unsqueeze(0),
        vocoder,
        model_config,
        preprocess_config,
    )[0]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(os.path.join(sample_dir, "{}_generated.wav".format(sample_id)), sampling_rate, prediction_audio)
    wavfile.write(os.path.join(sample_dir, "{}_gt.wav".format(sample_id)), sampling_rate, target_audio)


def build_loader(preprocess_config, train_config, max_samples):
    eval_config = deepcopy(train_config)
    eval_config["optimizer"] = dict(eval_config["optimizer"])
    eval_config["optimizer"]["batch_size"] = min(eval_config["optimizer"].get("batch_size", 16), max(max_samples, 1))
    dataset = Dataset("val.txt", preprocess_config, eval_config, sort=False, drop_last=False)
    if max_samples > 0:
        dataset.basename = dataset.basename[:max_samples]
        dataset.speaker = dataset.speaker[:max_samples]
        dataset.text = dataset.text[:max_samples]
        dataset.raw_text = dataset.raw_text[:max_samples]
    loader = DataLoader(
        dataset,
        batch_size=eval_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def sample_with_gt_duration(model, batch):
    return model(
        speakers=batch[2],
        texts=batch[3],
        src_lens=batch[4],
        max_src_len=batch[5],
        mels=None,
        mel_lens=batch[7],
        max_mel_len=batch[8],
        d_targets=batch[9],
    )


def evaluate_combination(
    model,
    loader,
    configs,
    restore_step,
    guidance_scale,
    sample_steps,
    seed,
    output_dir,
    save_samples,
    vocoder,
):
    preprocess_config, model_config, train_config = configs
    set_seed(seed)
    train_config.setdefault("mel_flow", {})["guidance_scale"] = guidance_scale
    train_config.setdefault("mel_flow", {})["sample_steps"] = sample_steps
    if hasattr(model, "set_train_config"):
        model.set_train_config(train_config)

    combo_name = "step{}_cfg{}_steps{}".format(
        restore_step,
        format_number(guidance_scale),
        sample_steps,
    )
    sample_dir = os.path.join(output_dir, "samples", combo_name)
    metrics = {
        "mel_l1": [],
        "mel_mse": [],
        "mel_rmse": [],
        "mel_mean_abs_diff": [],
        "mel_std_abs_diff": [],
    }
    saved_count = 0
    evaluated_count = 0

    model.eval()
    with torch.inference_mode():
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = sample_with_gt_duration(model, batch)
                mel_predictions = output[0]
                mel_targets = batch[6][:, : mel_predictions.shape[1], :]
                mel_lens = batch[7]
                for i, sample_id in enumerate(batch[0]):
                    mel_len = int(mel_lens[i].item())
                    sample_metrics = compute_mel_metrics(
                        mel_predictions[i],
                        mel_targets[i],
                        mel_len,
                    )
                    for key, value in sample_metrics.items():
                        metrics[key].append(value)
                    if saved_count < save_samples:
                        save_sample_artifacts(
                            sample_dir,
                            sample_id,
                            mel_predictions[i],
                            mel_targets[i],
                            mel_len,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        saved_count += 1
                    evaluated_count += 1

    row = {
        "restore_step": restore_step,
        "guidance_scale": guidance_scale,
        "sample_steps": sample_steps,
        "seed": seed,
        "num_samples": evaluated_count,
        "sample_dir": sample_dir if saved_count > 0 else "",
    }
    for key, values in metrics.items():
        summary = metric_summary(values)
        row["{}_mean".format(key)] = summary["mean"]
        row["{}_std".format(key)] = summary["std"]
    row["score"] = score_row(row)
    return row


def load_configs(args):
    with open(args.preprocess_config, "r", encoding="utf-8") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def run(args):
    ensure_dir(args.output_dir)
    base_configs = load_configs(args)
    preprocess_config, model_config, base_train_config = base_configs
    dataset, loader = build_loader(preprocess_config, base_train_config, args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")

    vocoder = None
    if args.save_wav:
        vocoder = get_vocoder(model_config, device)

    rows = []
    for restore_step in args.restore_steps:
        model_args = argparse.Namespace(restore_step=restore_step)
        train_config = deepcopy(base_train_config)
        configs = (preprocess_config, model_config, train_config)
        model = get_model(model_args, configs, device, train=False).to(device)
        if vocoder is not None:
            move_vocoder(vocoder, model_config, device)
        for guidance_scale in args.guidance_scales:
            for sample_steps in args.sample_steps:
                row = evaluate_combination(
                    model,
                    loader,
                    configs,
                    restore_step,
                    guidance_scale,
                    sample_steps,
                    args.seed,
                    args.output_dir,
                    args.save_samples,
                    vocoder,
                )
                rows.append(row)
                print(
                    "step={}, cfg={}, steps={}, mel_l1={:.6f}, std_diff={:.6f}, score={:.6f}".format(
                        restore_step,
                        guidance_scale,
                        sample_steps,
                        row["mel_l1_mean"],
                        row["mel_std_abs_diff_mean"],
                        row["score"],
                    )
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda row: row["score"])
    best = rows[0]
    write_csv(os.path.join(args.output_dir, "summary.csv"), rows)
    write_json(os.path.join(args.output_dir, "summary.json"), {"rows": rows})
    write_json(
        os.path.join(args.output_dir, "best.json"),
        {
            "best_by_score": best,
            "score_formula": "mel_l1_mean + 0.1 * mel_std_abs_diff_mean",
            "ranking_note": "Automatic mel metrics are for candidate selection; final choice should be confirmed by listening to generated samples.",
            "evaluation_protocol": "GT duration validation-set sampling; duration predictor error is intentionally excluded.",
        },
    )
    print("Best automatic combination:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return {"rows": rows, "best": best}


def self_test():
    pred = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [0.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.5, 1.0],
            [2.0, 5.0],
            [9.0, 9.0],
        ]
    )
    metrics = compute_mel_metrics(pred, target, 2)
    assert abs(metrics["mel_l1"] - 0.875) < 1e-6
    assert abs(metrics["mel_mse"] - 0.8125) < 1e-6
    summary = metric_summary([1.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["std"] == 1.0
    row = {"mel_l1_mean": 0.5, "mel_std_abs_diff_mean": 0.2}
    assert abs(score_row(row) - 0.52) < 1e-6
    assert split_ints("1,2,3") == [1, 2, 3]
    assert split_floats("1,1.5") == [1.0, 1.5]
    print("evaluate_flow_mel self_test ok")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate flow-mel checkpoints with GT duration")
    parser.add_argument("--self_test", action="store_true", help="Run helper-function self test and exit")
    parser.add_argument("--restore_steps", type=split_ints, default=[], help="Comma-separated checkpoint steps, e.g. 100000,200000")
    parser.add_argument("--guidance_scales", type=split_floats, default=split_floats("1.0,1.5,2.0"))
    parser.add_argument("--sample_steps", type=split_ints, default=split_ints("16,32,64"))
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--save_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_wav", action="store_true", help="Save generated/GT wavs with the configured vocoder")
    parser.add_argument("--output_dir", type=str, default="")
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
        if not parsed_args.restore_steps:
            missing.append("--restore_steps")
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
