import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from utils.model import get_model, move_vocoder
from utils.tools import close_figure, to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_model(model, batch, d_control=1.0):
    return model(
        speakers=batch[2],
        texts=batch[3],
        src_lens=batch[4],
        max_src_len=batch[5],
        mels=batch[6] if len(batch) > 6 else None,
        mel_lens=batch[7] if len(batch) > 7 else None,
        max_mel_len=batch[8] if len(batch) > 8 else None,
        d_targets=batch[9] if len(batch) > 9 else None,
        d_control=d_control,
    )


def evaluate(
    model,
    step,
    configs,
    logger=None,
    vocoder=None,
    loader=None,
    loss_fn=None,
    dataset=None,
):
    preprocess_config, model_config, train_config = configs

    if dataset is None:
        dataset = Dataset(
            "val.txt", preprocess_config, train_config, sort=False, drop_last=False
        )
    if loader is None:
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
    if loss_fn is None:
        loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    loss_sums = None
    sample_batch = None
    sample_output = None
    with torch.inference_mode():
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = forward_model(model, batch)
                losses = loss_fn(batch, output)
                if loss_sums is None:
                    loss_sums = [0 for _ in range(len(losses))]

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

                if sample_batch is None:
                    sample_batch = batch
                    sample_output = output

    if loss_sums is None:
        loss_sums = [0 for _ in range(4)]
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    if len(loss_means) > 8:
        message = "Validation Step {}, Total Loss: {:.4f}, Flow Mel Loss: {:.4f}, Duration Loss: {:.4f}, Phoneme Adv Loss: {:.4f}, VQ Loss: {:.4f}, VQ Commitment Loss: {:.4f}, VQ Codebook Loss: {:.4f}, Codebook Perplexity: {:.4f}, Used Codes: {:.1f}".format(
            *([step] + [l for l in loss_means])
        )
    elif len(loss_means) > 3:
        message = "Validation Step {}, Total Loss: {:.4f}, Flow Mel Loss: {:.4f}, Duration Loss: {:.4f}, Phoneme Adv Loss: {:.4f}".format(
            *([step] + [l for l in loss_means])
        )
    else:
        message = "Validation Step {}, Total Loss: {:.4f}, Flow Mel Loss: {:.4f}, Duration Loss: {:.4f}".format(
            *([step] + [l for l in loss_means])
        )

    if logger is not None:
        log(logger, step, losses=loss_means)

    if logger is not None and sample_batch is not None and sample_output is not None:
        logging_config = train_config.get("logging", {})
        log_audio = logging_config.get("log_audio", True)
        log_figure = logging_config.get("log_figure", True)
        active_vocoder = vocoder if log_audio else None
        model_device = next(model.parameters()).device
        if active_vocoder is not None:
            move_vocoder(active_vocoder, model_config, model_device)
        fig, wav_reconstruction, wav_prediction, _ = synth_one_sample(
            sample_batch,
            sample_output,
            active_vocoder,
            model_config,
            preprocess_config,
        )
        if active_vocoder is not None:
            move_vocoder(active_vocoder, model_config, torch.device("cpu"))

        if log_figure:
            log(
                logger,
                step,
                fig=fig,
                tag="Validation/mel",
            )
        else:
            close_figure(fig)
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        if log_audio:
            log(
                logger,
                step,
                audio=wav_reconstruction,
                sampling_rate=sampling_rate,
                tag="Validation/reconstructed",
            )
            log(
                logger,
                step,
                audio=wav_prediction,
                sampling_rate=sampling_rate,
                tag="Validation/synthesized",
            )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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

    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
