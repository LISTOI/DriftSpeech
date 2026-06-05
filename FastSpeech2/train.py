import argparse
import json
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, move_vocoder
from utils.tools import close_figure, to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate, forward_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_stylecodes(model, val_loader, val_dataset, device, train_config, step):
    config = train_config.get("stylecode_logging", {})
    if not config.get("enabled", True):
        return
    fs2_model = model.module if hasattr(model, "module") else model
    if hasattr(fs2_model, "mel_flow"):
        return

    num_samples = min(config.get("num_samples", 128), len(val_dataset))
    if num_samples <= 0:
        return

    output_dir = os.path.join(
        train_config["path"]["result_path"],
        config.get("dir_name", "stylecode"),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "{}.jsonl".format(step))
    include_token_ids = config.get("include_token_ids", True)
    include_phoneme_text = config.get("include_phoneme_text", True)
    phoneme_by_id = (
        dict(zip(val_dataset.basename, val_dataset.text)) if include_phoneme_text else {}
    )

    was_training = model.training
    model.eval()
    records_written = 0

    with torch.inference_mode(), open(output_path, "w", encoding="utf-8") as f:
        for batchs in val_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                stylecodes, style_info = fs2_model.style_extractor.extract_stylecode_with_info(
                    batch[6],
                    batch[9],
                    src_lens=batch[4],
                    mel_lens=batch[7],
                    include_continuous=True,
                )
                for i, sample_id in enumerate(batch[0]):
                    if records_written >= num_samples:
                        break
                    src_len = int(batch[4][i].item())
                    mel_len = int(batch[7][i].item())
                    record = {
                        "step": step,
                        "id": sample_id,
                        "speaker": int(batch[2][i].item()),
                        "text": batch[1][i],
                        "src_len": src_len,
                        "mel_len": mel_len,
                        "bottleneck_dim": fs2_model.style_extractor.bottleneck_dim,
                        "durations": batch[9][i, :src_len].detach().cpu().tolist(),
                        "stylecode_shape": [src_len, fs2_model.style_extractor.bottleneck_dim],
                        "stylecode": stylecodes[i, :src_len].detach().cpu().tolist(),
                    }
                    if style_info is not None:
                        if "continuous_stylecode" in style_info:
                            continuous_stylecode = style_info["continuous_stylecode"]
                            record["continuous_stylecode_shape"] = [src_len, fs2_model.style_extractor.bottleneck_dim]
                            record["continuous_stylecode"] = continuous_stylecode[i, :src_len].detach().cpu().tolist()
                        if "code_indices" in style_info:
                            record["vq_indices"] = style_info["code_indices"][i, :src_len].detach().cpu().tolist()
                        record["codebook_size"] = int(style_info.get("codebook_size", 0))
                        if "codebook_perplexity" in style_info:
                            record["codebook_perplexity"] = float(style_info["codebook_perplexity"].detach().cpu().item())
                        if "used_code_count" in style_info:
                            record["used_code_count"] = float(style_info["used_code_count"].detach().cpu().item())
                    if include_token_ids:
                        record["token_ids"] = batch[3][i, :src_len].detach().cpu().tolist()
                    if include_phoneme_text:
                        record["phoneme_text"] = phoneme_by_id.get(sample_id, "")
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
                if records_written >= num_samples:
                    break
            if records_written >= num_samples:
                break

    if was_training:
        model.train()


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config.get("data", {}).get("group_size", 4)
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    val_loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    logging_config = train_config.get("logging", {})
    log_audio = logging_config.get("log_audio", True)
    log_figure = logging_config.get("log_figure", True)
    vocoder = get_vocoder(model_config, torch.device("cpu")) if log_audio else None

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    try:
        while True:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
            for batchs in loader:
                for batch in batchs:
                    batch = to_device(batch, device)

                    output = forward_model(model, batch)

                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    total_loss = total_loss / grad_acc_step
                    total_loss.backward()
                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                    if step % log_step == 0:
                        loss_values = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Flow Mel Loss: {:.4f}, Duration Loss: {:.4f}".format(
                            *loss_values
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)
                        log(train_logger, step, losses=loss_values)

                    if step % synth_step == 0 and (log_figure or log_audio):
                        if vocoder is not None:
                            move_vocoder(vocoder, model_config, device)
                        fig, wav_reconstruction, wav_prediction, _ = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        if vocoder is not None:
                            move_vocoder(vocoder, model_config, torch.device("cpu"))

                        if log_figure:
                            log(
                                train_logger,
                                step,
                                fig=fig,
                                tag="Training/mel",
                            )
                        else:
                            close_figure(fig)
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        if log_audio:
                            log(
                                train_logger,
                                step,
                                audio=wav_reconstruction,
                                sampling_rate=sampling_rate,
                                tag="Training/reconstructed",
                            )
                            log(
                                train_logger,
                                step,
                                audio=wav_prediction,
                                sampling_rate=sampling_rate,
                                tag="Training/synthesized",
                            )

                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(
                            model,
                            step,
                            configs,
                            val_logger,
                            vocoder,
                            loader=val_loader,
                            loss_fn=val_loss,
                            dataset=val_dataset,
                        )
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)
                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )
                        dump_stylecodes(
                            model,
                            val_loader,
                            val_dataset,
                            device,
                            train_config,
                            step,
                        )

                    if step == total_step:
                        return
                    step += 1
                    outer_bar.update(1)

                inner_bar.update(1)
            epoch += 1
    finally:
        train_logger.flush()
        val_logger.flush()
        train_logger.close()
        val_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    with open(args.preprocess_config, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
