import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_model_state(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        optional_prefixes = (
            "style_extractor.phoneme_adversary.",
            "style_extractor.vector_quantizer.",
        )
        model_state = model.state_dict()
        filtered_state = {}
        skipped_optional = []
        for key, value in state_dict.items():
            if key.startswith(optional_prefixes):
                if key not in model_state:
                    skipped_optional.append(key)
                    continue
                if model_state[key].shape != value.shape:
                    skipped_optional.append(key)
                    continue
            filtered_state[key] = value

        incompatible = model.load_state_dict(filtered_state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing = all(
            key.startswith(optional_prefixes) for key in missing
        )
        allowed_unexpected = all(
            key.startswith(optional_prefixes) for key in unexpected
        )
        if not allowed_missing or not allowed_unexpected:
            raise exc
        if missing:
            print(
                "Loaded checkpoint with newly initialized optional style parameters: {}".format(
                    missing
                )
            )
        if skipped_optional or unexpected:
            print(
                "Ignored optional style parameters from checkpoint: {}".format(
                    skipped_optional + unexpected
                )
            )


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = load_checkpoint(ckpt_path, device)
        _load_model_state(model, ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            try:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print(
                    "Optimizer state is incompatible with optional style parameters; starting a fresh optimizer state."
                )
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_(False)
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = load_checkpoint("hifigan/generator_LJSpeech.pth.tar", device)
        elif speaker == "universal":
            ckpt = load_checkpoint("hifigan/generator_universal.pth.tar", device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def get_vocoder_device(vocoder, model_config):
    name = model_config["vocoder"]["model"]
    if name == "MelGAN":
        return next(vocoder.mel2wav.parameters()).device
    if name == "HiFi-GAN":
        return next(vocoder.parameters()).device
    return torch.device("cpu")


def move_vocoder(vocoder, model_config, device):
    if vocoder is None:
        return None
    name = model_config["vocoder"]["model"]
    if name == "MelGAN":
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    vocoder_device = get_vocoder_device(vocoder, model_config)
    mels = mels.to(vocoder_device)
    with torch.inference_mode():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            length = lengths[i]
            if hasattr(length, "item"):
                length = length.item()
            wavs[i] = wavs[i][: int(length)]

    return wavs
