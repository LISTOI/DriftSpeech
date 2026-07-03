import argparse
import json
import os
from copy import deepcopy
from pathlib import Path


def split_names(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one split name")
    return items


def split_label(split_name):
    return Path(split_name).stem.replace(os.sep, "_")


def output_path_for_split(output_dir, restore_step, split_name):
    return os.path.join(output_dir, "{}_{}.jsonl".format(restore_step, split_label(split_name)))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_record(record):
    src_len = int(record["src_len"])
    bottleneck_dim = int(record["bottleneck_dim"])
    if len(record["durations"]) != src_len:
        raise ValueError("durations length must equal src_len")
    if "token_ids" in record and len(record["token_ids"]) != src_len:
        raise ValueError("token_ids length must equal src_len")
    if record["stylecode_shape"] != [src_len, bottleneck_dim]:
        raise ValueError("stylecode_shape must equal [src_len, bottleneck_dim]")
    if len(record["stylecode"]) != src_len:
        raise ValueError("stylecode length must equal src_len")
    for row in record["stylecode"]:
        if len(row) != bottleneck_dim:
            raise ValueError("each stylecode row length must equal bottleneck_dim")
    if "continuous_stylecode" in record:
        if record.get("continuous_stylecode_shape") != [src_len, bottleneck_dim]:
            raise ValueError("continuous_stylecode_shape must equal [src_len, bottleneck_dim]")
        if len(record["continuous_stylecode"]) != src_len:
            raise ValueError("continuous_stylecode length must equal src_len")
        for row in record["continuous_stylecode"]:
            if len(row) != bottleneck_dim:
                raise ValueError("each continuous_stylecode row length must equal bottleneck_dim")
    if "vq_indices" in record and len(record["vq_indices"]) != src_len:
        raise ValueError("vq_indices length must equal src_len")


def load_configs(args):
    import yaml

    with open(args.preprocess_config, "r", encoding="utf-8") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def build_loader(split_name, preprocess_config, train_config, batch_size):
    from torch.utils.data import DataLoader

    from dataset import Dataset

    export_config = deepcopy(train_config)
    export_config["optimizer"] = dict(export_config["optimizer"])
    export_config["optimizer"]["batch_size"] = batch_size
    dataset = Dataset(split_name, preprocess_config, export_config, sort=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def build_record(
    split_name,
    restore_step,
    batch,
    sample_index,
    stylecodes,
    style_info,
    bottleneck_dim,
    phoneme_by_id,
    include_token_ids=True,
    include_phoneme_text=True,
):
    sample_id = batch[0][sample_index]
    src_len = int(batch[4][sample_index].item())
    mel_len = int(batch[7][sample_index].item())
    record = {
        "step": restore_step,
        "split": split_label(split_name),
        "id": sample_id,
        "speaker": int(batch[2][sample_index].item()),
        "text": batch[1][sample_index],
        "src_len": src_len,
        "mel_len": mel_len,
        "bottleneck_dim": bottleneck_dim,
        "durations": batch[9][sample_index, :src_len].detach().cpu().tolist(),
        "stylecode_shape": [src_len, bottleneck_dim],
        "stylecode": stylecodes[sample_index, :src_len].detach().cpu().tolist(),
    }
    if style_info is not None:
        if "continuous_stylecode" in style_info:
            continuous_stylecode = style_info["continuous_stylecode"]
            record["continuous_stylecode_shape"] = [src_len, bottleneck_dim]
            record["continuous_stylecode"] = continuous_stylecode[sample_index, :src_len].detach().cpu().tolist()
        if "code_indices" in style_info:
            record["vq_indices"] = style_info["code_indices"][sample_index, :src_len].detach().cpu().tolist()
        if "codebook_size" in style_info:
            record["codebook_size"] = int(style_info.get("codebook_size", 0))
        if "codebook_perplexity" in style_info:
            record["codebook_perplexity"] = float(style_info["codebook_perplexity"].detach().cpu().item())
        if "used_code_count" in style_info:
            record["used_code_count"] = float(style_info["used_code_count"].detach().cpu().item())
    if include_token_ids:
        record["token_ids"] = batch[3][sample_index, :src_len].detach().cpu().tolist()
    if include_phoneme_text:
        record["phoneme_text"] = phoneme_by_id.get(sample_id, "")
    validate_record(record)
    return record


def export_split(model, split_name, preprocess_config, train_config, args):
    import torch
    from tqdm import tqdm

    from utils.tools import to_device

    fs2_model = model.module if hasattr(model, "module") else model
    if not hasattr(fs2_model, "style_extractor"):
        raise RuntimeError("Loaded model does not have style_extractor")

    dataset, loader = build_loader(split_name, preprocess_config, train_config, args.batch_size)
    if len(dataset) == 0:
        raise RuntimeError("Dataset split is empty: {}".format(split_name))
    phoneme_by_id = dict(zip(dataset.basename, dataset.text)) if args.include_phoneme_text else {}
    output_path = output_path_for_split(args.output_dir, args.restore_step, split_name)
    ensure_dir(os.path.dirname(output_path))

    was_training = model.training
    model.eval()
    records_written = 0
    bottleneck_dim = int(fs2_model.style_extractor.bottleneck_dim)
    with torch.inference_mode(), open(output_path, "w", encoding="utf-8") as f:
        progress = tqdm(loader, desc="export {}".format(split_label(split_name)))
        for batchs in progress:
            for batch in batchs:
                batch = to_device(batch, args.device)
                stylecodes, style_info = fs2_model.style_extractor.extract_stylecode_with_info(
                    batch[6],
                    batch[9],
                    src_lens=batch[4],
                    mel_lens=batch[7],
                    include_continuous=args.include_continuous,
                )
                for sample_index in range(len(batch[0])):
                    record = build_record(
                        split_name,
                        args.restore_step,
                        batch,
                        sample_index,
                        stylecodes,
                        style_info,
                        bottleneck_dim,
                        phoneme_by_id,
                        include_token_ids=args.include_token_ids,
                        include_phoneme_text=args.include_phoneme_text,
                    )
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
    if was_training:
        model.train()
    return {
        "split": split_label(split_name),
        "input_file": split_name,
        "num_records": records_written,
        "output_path": output_path,
        "bottleneck_dim": bottleneck_dim,
    }


def default_output_dir(train_config):
    return os.path.join(train_config["path"]["result_path"], "stylecode_predictor_data")


def run(args):
    import torch

    from utils.model import get_model

    preprocess_config, model_config, train_config = load_configs(args)
    if not args.output_dir:
        args.output_dir = default_output_dir(train_config)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_args = argparse.Namespace(restore_step=args.restore_step)
    model = get_model(model_args, (preprocess_config, model_config, train_config), args.device, train=False).to(args.device)

    ensure_dir(args.output_dir)
    split_summaries = []
    for split_name in args.splits:
        split_summaries.append(export_split(model, split_name, preprocess_config, train_config, args))

    manifest_path = os.path.join(args.output_dir, "{}_manifest.json".format(args.restore_step))
    manifest = {
        "restore_step": args.restore_step,
        "output_dir": args.output_dir,
        "preprocess_config": args.preprocess_config,
        "model_config": args.model_config,
        "train_config": args.train_config,
        "include_token_ids": args.include_token_ids,
        "include_phoneme_text": args.include_phoneme_text,
        "include_continuous": args.include_continuous,
        "splits": split_summaries,
        "manifest_path": manifest_path,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def self_test():
    assert split_names("train.txt,val.txt") == ["train.txt", "val.txt"]
    assert output_path_for_split("/tmp/out", 900000, "train.txt").endswith("900000_train.jsonl")
    record = {
        "src_len": 2,
        "bottleneck_dim": 3,
        "durations": [1, 2],
        "token_ids": [10, 20],
        "stylecode_shape": [2, 3],
        "stylecode": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }
    validate_record(record)
    bad_record = dict(record)
    bad_record["token_ids"] = [10]
    try:
        validate_record(bad_record)
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("validate_record should reject token_ids length mismatch")
    print("export_stylecodes self_test ok")


def parse_args():
    parser = argparse.ArgumentParser("Export full stylecode JSONL files for StyleCodePredictor")
    parser.add_argument("--self_test", action="store_true", help="Run helper self test and exit")
    parser.add_argument("--restore_step", type=int, default=0, help="FastSpeech2 checkpoint step to load")
    parser.add_argument("--splits", type=split_names, default=split_names("train.txt,val.txt"), help="Comma-separated metadata files to export")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory; defaults to result_path/stylecode_predictor_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--include_token_ids", action="store_true", default=True)
    parser.add_argument("--no_include_token_ids", dest="include_token_ids", action="store_false")
    parser.add_argument("--include_phoneme_text", action="store_true", default=True)
    parser.add_argument("--no_include_phoneme_text", dest="include_phoneme_text", action="store_false")
    parser.add_argument("--include_continuous", action="store_true", default=True)
    parser.add_argument("--no_include_continuous", dest="include_continuous", action="store_false")
    parser.add_argument("--cpu", action="store_true", help="Force CPU export")
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
        if not parsed_args.restore_step:
            missing.append("--restore_step")
        if not parsed_args.preprocess_config:
            missing.append("--preprocess_config")
        if not parsed_args.model_config:
            missing.append("--model_config")
        if not parsed_args.train_config:
            missing.append("--train_config")
        if missing:
            raise SystemExit("Missing required arguments: {}".format(", ".join(missing)))
        run(parsed_args)
