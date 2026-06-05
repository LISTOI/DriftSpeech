#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def validate_record(record, path, line_number, require_token_ids):
    src_len = int(record.get("src_len", 0))
    if src_len <= 0:
        raise ValueError(f"{path}:{line_number} invalid src_len={src_len}")

    stylecode = record.get("stylecode")
    if not isinstance(stylecode, list) or len(stylecode) != src_len:
        raise ValueError(f"{path}:{line_number} invalid stylecode length")
    if not stylecode or not isinstance(stylecode[0], list):
        raise ValueError(f"{path}:{line_number} stylecode must be a 2D list")
    style_dim = len(stylecode[0])
    for row in stylecode:
        if not isinstance(row, list) or len(row) != style_dim:
            raise ValueError(f"{path}:{line_number} stylecode must be rectangular")

    shape = record.get("stylecode_shape")
    if shape is not None and list(shape) != [src_len, style_dim]:
        raise ValueError(f"{path}:{line_number} stylecode_shape mismatch")
    if "bottleneck_dim" in record and int(record["bottleneck_dim"]) != style_dim:
        raise ValueError(f"{path}:{line_number} bottleneck_dim mismatch")

    token_ids = record.get("token_ids")
    if require_token_ids and token_ids is None:
        raise ValueError(f"{path}:{line_number} missing token_ids")
    max_token_id = None
    if token_ids is not None:
        if len(token_ids) != src_len:
            raise ValueError(f"{path}:{line_number} token_ids length mismatch")
        max_token_id = max([int(token) for token in token_ids], default=0)

    durations = record.get("durations")
    if durations is None:
        raise ValueError(f"{path}:{line_number} missing durations")
    if len(durations) != src_len:
        raise ValueError(f"{path}:{line_number} durations length mismatch")
    durations = [int(duration) for duration in durations]

    return {
        "src_len": src_len,
        "style_dim": style_dim,
        "max_token_id": max_token_id,
        "zero_durations": sum(1 for duration in durations if duration <= 0),
        "duration_count": len(durations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--require-token-ids", action="store_true")
    args = parser.parse_args()

    num_records = 0
    src_lens = []
    style_dims = Counter()
    max_token_id = 0
    zero_durations = 0
    duration_count = 0

    for jsonl_path in args.jsonl:
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                stats = validate_record(json.loads(line), path, line_number, args.require_token_ids)
                num_records += 1
                src_lens.append(stats["src_len"])
                style_dims[stats["style_dim"]] += 1
                if stats["max_token_id"] is not None:
                    max_token_id = max(max_token_id, stats["max_token_id"])
                zero_durations += stats["zero_durations"]
                duration_count += stats["duration_count"]

    if num_records == 0:
        raise RuntimeError("No records found")

    print(json.dumps({
        "num_records": num_records,
        "src_len_min": min(src_lens),
        "src_len_max": max(src_lens),
        "src_len_mean": sum(src_lens) / len(src_lens),
        "style_dim_distribution": dict(style_dims),
        "max_token_id": max_token_id,
        "zero_duration_rate": zero_durations / max(duration_count, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
