import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .masks import lengths_to_padding_mask


class StyleCodeJsonlDataset(Dataset):
    def __init__(self, jsonl_paths, require_token_ids=True, require_durations=True):
        if isinstance(jsonl_paths, (str, Path)):
            jsonl_paths = [jsonl_paths]
        self.paths = [Path(path) for path in jsonl_paths]
        self.require_token_ids = require_token_ids
        self.require_durations = require_durations
        self.records = []
        self.style_dim = None
        self.max_token_id = 0
        for path in self.paths:
            self._load_path(path)
        if not self.records:
            raise ValueError("No stylecode records were loaded")

    def _load_path(self, path):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample = self._parse_record(record, path, line_number)
                self.records.append(sample)

    def _parse_record(self, record, path, line_number):
        src_len = int(record.get("src_len", 0))
        if src_len <= 0:
            raise ValueError(f"{path}:{line_number} has invalid src_len={src_len}")

        stylecode = record.get("stylecode")
        if not isinstance(stylecode, list) or len(stylecode) != src_len:
            raise ValueError(f"{path}:{line_number} stylecode length does not match src_len")
        if not stylecode or not isinstance(stylecode[0], list):
            raise ValueError(f"{path}:{line_number} stylecode must be a 2D list")
        style_dim = len(stylecode[0])
        if style_dim <= 0:
            raise ValueError(f"{path}:{line_number} stylecode dimension must be positive")
        for row in stylecode:
            if not isinstance(row, list) or len(row) != style_dim:
                raise ValueError(f"{path}:{line_number} stylecode must be rectangular")

        shape = record.get("stylecode_shape")
        if shape is not None and list(shape) != [src_len, style_dim]:
            raise ValueError(f"{path}:{line_number} stylecode_shape does not match actual shape")
        bottleneck_dim = record.get("bottleneck_dim")
        if bottleneck_dim is not None and int(bottleneck_dim) != style_dim:
            raise ValueError(f"{path}:{line_number} bottleneck_dim does not match stylecode")
        if self.style_dim is None:
            self.style_dim = style_dim
        elif self.style_dim != style_dim:
            raise ValueError(f"{path}:{line_number} style_dim={style_dim} differs from {self.style_dim}")

        token_ids = record.get("token_ids")
        if token_ids is None:
            if self.require_token_ids:
                raise ValueError(f"{path}:{line_number} is missing token_ids")
            token_ids = [0] * src_len
        if len(token_ids) != src_len:
            raise ValueError(f"{path}:{line_number} token_ids length does not match src_len")
        token_ids = [int(token) for token in token_ids]
        if token_ids:
            self.max_token_id = max(self.max_token_id, max(token_ids))

        durations = record.get("durations")
        if durations is None:
            if self.require_durations:
                raise ValueError(f"{path}:{line_number} is missing durations")
            durations = [1] * src_len
        if len(durations) != src_len:
            raise ValueError(f"{path}:{line_number} durations length does not match src_len")
        durations = [int(duration) for duration in durations]

        speaker = record.get("speaker", 0)
        try:
            speaker = int(speaker)
        except (TypeError, ValueError):
            speaker = 0

        return {
            "id": str(record.get("id", f"{path.stem}:{line_number}")),
            "speaker": speaker,
            "text": str(record.get("text", "")),
            "phoneme_text": str(record.get("phoneme_text", "")),
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "stylecode": torch.tensor(stylecode, dtype=torch.float32),
            "durations": torch.tensor(durations, dtype=torch.long),
            "length": src_len,
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_stylecode_batch(samples, pad_token_id=0):
    batch_size = len(samples)
    max_len = max(sample["length"] for sample in samples)
    style_dim = samples[0]["stylecode"].shape[1]

    tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    stylecode = torch.zeros(batch_size, max_len, style_dim, dtype=torch.float32)
    durations = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)

    for i, sample in enumerate(samples):
        length = sample["length"]
        tokens[i, :length] = sample["tokens"]
        stylecode[i, :length] = sample["stylecode"]
        durations[i, :length] = sample["durations"]

    return {
        "ids": [sample["id"] for sample in samples],
        "speakers": [sample["speaker"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
        "phoneme_texts": [sample["phoneme_text"] for sample in samples],
        "tokens": tokens,
        "stylecode": stylecode,
        "durations": durations,
        "lengths": lengths,
        "padding_mask": lengths_to_padding_mask(lengths, max_len=max_len),
    }


def make_collate_fn(pad_token_id=0):
    return partial(collate_stylecode_batch, pad_token_id=pad_token_id)


def infer_dataset_metadata(dataset):
    return {
        "style_dim": dataset.style_dim,
        "max_token_id": dataset.max_token_id,
        "num_records": len(dataset),
    }
