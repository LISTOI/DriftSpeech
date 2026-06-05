from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SILENCE_LABELS = {"sp", "spn", "sil"}
DEPENDENCY_HELP = (
    "Missing dependency '{package}'. Install evaluation dependencies with: "
    "conda run -n torch_gpu pip install -r /home/listoi/Speech/stylecoode/evaluate/requirements.txt"
)


def require_dependency(package: str, import_name: str | None = None):
    try:
        module = __import__(import_name or package, fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(DEPENDENCY_HELP.format(package=package)) from exc
    return module


@dataclass
class ParsedUtterance:
    vectors: np.ndarray
    labels: list[str]
    raw_labels: list[str]
    token_ids: list[Any]
    durations: list[Any]
    code_indices: list[Any]
    codebook_size: Any
    sample_id: str
    step: Any
    speaker: Any
    source: str


@dataclass
class Point:
    vector: np.ndarray
    label: str
    raw_label: str
    sample_id: str
    position: int
    step: Any
    speaker: Any
    token_id: Any
    duration: Any
    code_index: Any
    codebook_size: Any


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_label(label: str, keep_stress: bool) -> str:
    label = label.strip()
    if not keep_stress and re.fullmatch(r"[A-Z]+[012]", label):
        return label[:-1]
    return label


def parse_phoneme_text(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip().replace("{", " ").replace("}", " ")
    return [part for part in text.split() if part]


def load_token_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("--token_id_to_label_json must point to a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def should_exclude_label(label: str, args: argparse.Namespace, excluded: set[str]) -> bool:
    if not args.include_silence and label.lower() in SILENCE_LABELS:
        return True
    return label in excluded or label.lower() in excluded


def parse_code_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    if code < 0:
        return None
    return code


def handle_parse_error(
    message: str,
    path: Path,
    line_no: int,
    args: argparse.Namespace,
    stats: Counter,
    reason: str,
) -> None:
    if args.fail_on_mismatch:
        raise ValueError(f"{path}:{line_no}: {message}")
    stats[reason] += 1


def parse_record(
    record: dict[str, Any],
    path: Path,
    line_no: int,
    args: argparse.Namespace,
    token_map: dict[str, str],
    stats: Counter,
) -> ParsedUtterance | None:
    if "stylecode" not in record:
        handle_parse_error("missing stylecode", path, line_no, args, stats, "missing_stylecode")
        return None

    try:
        vectors = np.asarray(record["stylecode"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        handle_parse_error(f"invalid stylecode: {exc}", path, line_no, args, stats, "invalid_stylecode")
        return None

    if vectors.ndim != 2:
        handle_parse_error(f"stylecode must be 2-D, got shape={vectors.shape}", path, line_no, args, stats, "invalid_stylecode_shape")
        return None
    if not np.isfinite(vectors).all():
        handle_parse_error("stylecode contains NaN or Inf", path, line_no, args, stats, "nonfinite_stylecode")
        return None

    src_len = int(record.get("src_len", vectors.shape[0]))
    if vectors.shape[0] != src_len:
        handle_parse_error(
            f"stylecode length {vectors.shape[0]} does not match src_len {src_len}",
            path,
            line_no,
            args,
            stats,
            "stylecode_src_len_mismatch",
        )
        return None

    stylecode_shape = record.get("stylecode_shape")
    if isinstance(stylecode_shape, list) and len(stylecode_shape) >= 2:
        if int(stylecode_shape[0]) != src_len or int(stylecode_shape[1]) != vectors.shape[1]:
            handle_parse_error(
                f"stylecode_shape {stylecode_shape} does not match actual shape {list(vectors.shape)}",
                path,
                line_no,
                args,
                stats,
                "stylecode_shape_mismatch",
            )
            return None

    phonemes = parse_phoneme_text(record.get("phoneme_text", ""))
    token_ids = record.get("token_ids", [])
    if not isinstance(token_ids, list):
        token_ids = []

    if len(phonemes) == src_len:
        raw_labels = phonemes
        source = "phoneme_text"
    elif len(token_ids) >= src_len:
        raw_labels = [token_map.get(str(token_id), f"token:{token_id}") for token_id in token_ids[:src_len]]
        source = "token_ids"
    else:
        handle_parse_error(
            f"cannot align labels: phoneme_count={len(phonemes)}, token_count={len(token_ids)}, src_len={src_len}",
            path,
            line_no,
            args,
            stats,
            "label_alignment_mismatch",
        )
        return None

    durations = record.get("durations", [])
    if not isinstance(durations, list):
        durations = []
    if len(durations) < src_len:
        durations = [*durations, *([None] * (src_len - len(durations)))]
    else:
        durations = durations[:src_len]

    code_indices = record.get("vq_indices", [])
    if not isinstance(code_indices, list):
        code_indices = []
    if len(code_indices) < src_len:
        code_indices = [*code_indices, *([None] * (src_len - len(code_indices)))]
    else:
        code_indices = code_indices[:src_len]
    if any(parse_code_index(code) is not None for code in code_indices):
        stats["records_with_vq_indices"] += 1

    token_ids = token_ids[:src_len] if len(token_ids) >= src_len else [None] * src_len
    normalized = [normalize_label(label, args.keep_stress) for label in raw_labels]
    stats[f"label_source_{source}"] += 1

    return ParsedUtterance(
        vectors=vectors,
        labels=normalized,
        raw_labels=raw_labels,
        token_ids=token_ids,
        durations=durations,
        code_indices=code_indices,
        codebook_size=record.get("codebook_size", None),
        sample_id=str(record.get("id", "")),
        step=record.get("step", ""),
        speaker=record.get("speaker", ""),
        source=source,
    )


def iter_utterances(
    paths: list[Path],
    args: argparse.Namespace,
    token_map: dict[str, str],
    stats: Counter,
):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                stats["lines_read"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    handle_parse_error(f"invalid JSON: {exc}", path, line_no, args, stats, "invalid_json")
                    continue
                if not isinstance(record, dict):
                    handle_parse_error("JSON line must be an object", path, line_no, args, stats, "invalid_record")
                    continue
                parsed = parse_record(record, path, line_no, args, token_map, stats)
                if parsed is None:
                    continue
                stats["records_parsed"] += 1
                yield parsed


def iter_valid_positions(
    utterance: ParsedUtterance,
    args: argparse.Namespace,
    excluded: set[str],
    stats: Counter,
):
    for position, label in enumerate(utterance.labels):
        duration = utterance.durations[position]
        if duration is not None and duration <= 0:
            stats["skipped_zero_duration"] += 1
            continue
        if should_exclude_label(label, args, excluded):
            stats["skipped_excluded_label"] += 1
            continue
        stats["valid_phoneme_positions"] += 1
        yield position, label


def count_labels(
    paths: list[Path],
    args: argparse.Namespace,
    token_map: dict[str, str],
    excluded: set[str],
) -> tuple[Counter, Counter, Counter]:
    label_counts: Counter = Counter()
    stats: Counter = Counter()
    dim_counts: Counter = Counter()

    for utterance in iter_utterances(paths, args, token_map, stats):
        dim_counts[int(utterance.vectors.shape[1])] += 1
        for _, label in iter_valid_positions(utterance, args, excluded, stats):
            label_counts[label] += 1

    return label_counts, dim_counts, stats


def choose_labels(label_counts: Counter, args: argparse.Namespace, excluded: set[str]) -> tuple[list[str], list[str]]:
    explicit = split_csv(args.labels)
    if explicit:
        selected = []
        missing = []
        for label in explicit:
            normalized = normalize_label(label, args.keep_stress)
            if should_exclude_label(normalized, args, excluded):
                continue
            if label_counts.get(normalized, 0) > 0:
                selected.append(normalized)
            else:
                missing.append(normalized)
        return selected, missing

    selected = [
        label
        for label, count in label_counts.most_common()
        if count >= args.min_count and not should_exclude_label(label, args, excluded)
    ]
    return selected[: args.top_k], []


def collect_candidate_points(
    paths: list[Path],
    args: argparse.Namespace,
    token_map: dict[str, str],
    excluded: set[str],
    selected_labels: list[str],
    expected_dim: int,
) -> tuple[dict[str, list[Point]], list[Point], Counter]:
    grouped: dict[str, list[Point]] = defaultdict(list)
    other_points: list[Point] = []
    stats: Counter = Counter()
    selected = set(selected_labels)

    for utterance in iter_utterances(paths, args, token_map, stats):
        if int(utterance.vectors.shape[1]) != expected_dim:
            stats["skipped_dim_mismatch"] += 1
            continue
        for position, label in iter_valid_positions(utterance, args, excluded, stats):
            point = Point(
                vector=utterance.vectors[position],
                label=label if label in selected else "other",
                raw_label=utterance.raw_labels[position],
                sample_id=utterance.sample_id,
                position=position,
                step=utterance.step,
                speaker=utterance.speaker,
                token_id=utterance.token_ids[position],
                duration=utterance.durations[position],
                code_index=utterance.code_indices[position],
                codebook_size=utterance.codebook_size,
            )
            if label in selected:
                grouped[label].append(point)
            elif args.include_other:
                other_points.append(point)

    return grouped, other_points, stats


def sample_points(
    grouped: dict[str, list[Point]],
    other_points: list[Point],
    selected_labels: list[str],
    args: argparse.Namespace,
) -> tuple[list[Point], Counter]:
    rng = np.random.default_rng(args.seed)
    sampled: list[Point] = []
    sampled_counts: Counter = Counter()
    per_label_limit = min(args.max_per_label, max(1, args.max_points // max(1, len(selected_labels))))

    for label in selected_labels:
        points = grouped.get(label, [])
        if len(points) > per_label_limit:
            indices = rng.choice(len(points), size=per_label_limit, replace=False)
            chosen = [points[int(index)] for index in indices]
        else:
            chosen = list(points)
        sampled.extend(chosen)
        sampled_counts[label] = len(chosen)

    remaining = max(0, args.max_points - len(sampled))
    if args.include_other and remaining > 0 and other_points:
        limit = min(args.max_other, remaining, len(other_points))
        if len(other_points) > limit:
            indices = rng.choice(len(other_points), size=limit, replace=False)
            chosen_other = [other_points[int(index)] for index in indices]
        else:
            chosen_other = list(other_points)
        sampled.extend(chosen_other)
        sampled_counts["other"] = len(chosen_other)

    permutation = rng.permutation(len(sampled))
    sampled = [sampled[int(index)] for index in permutation]
    return sampled, sampled_counts


def preprocess_features(features: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return features.astype(np.float32, copy=True)
    preprocessing = require_dependency("scikit-learn", "sklearn.preprocessing")
    if normalize == "standard":
        return preprocessing.StandardScaler().fit_transform(features).astype(np.float32)
    if normalize == "l2":
        return preprocessing.normalize(features).astype(np.float32)
    raise ValueError(f"Unsupported normalization: {normalize}")


def reduce_tsne(features: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    decomposition = require_dependency("scikit-learn", "sklearn.decomposition")
    manifold = require_dependency("scikit-learn", "sklearn.manifold")

    if features.shape[0] < 3:
        raise RuntimeError("At least 3 sampled points are required for t-SNE.")

    reduced = features
    pca_components = None
    if args.pca_dim > 0 and features.shape[1] > args.pca_dim:
        pca_components = min(args.pca_dim, features.shape[1], features.shape[0] - 1)
        reduced = decomposition.PCA(n_components=pca_components, random_state=args.seed).fit_transform(features)

    perplexity = min(float(args.perplexity), max(2.0, (features.shape[0] - 1) / 3.0))
    if perplexity >= features.shape[0]:
        perplexity = max(1.0, features.shape[0] - 1.0)

    init = "pca" if reduced.shape[1] >= 2 else "random"
    kwargs = {
        "n_components": 2,
        "random_state": args.seed,
        "init": init,
        "perplexity": perplexity,
        "learning_rate": "auto",
    }

    try:
        coords = manifold.TSNE(**kwargs).fit_transform(reduced)
    except (TypeError, ValueError):
        kwargs["learning_rate"] = 200.0
        coords = manifold.TSNE(**kwargs).fit_transform(reduced)

    return coords.astype(np.float32), {
        "pca_components": pca_components,
        "perplexity": perplexity,
        "tsne_init": init,
        "tsne_learning_rate": kwargs["learning_rate"],
    }


def build_palette(labels: list[str]) -> dict[str, Any]:
    sns = require_dependency("seaborn")

    non_other = [label for label in labels if label != "other"]
    colors = sns.color_palette("tab20", n_colors=max(1, len(non_other)))
    palette = {label: colors[index] for index, label in enumerate(non_other)}
    if "other" in labels:
        palette["other"] = "#BBBBBB"
    return palette


def plot_points(
    coords: np.ndarray,
    labels: list[str],
    hue_order: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    plt = require_dependency("matplotlib", "matplotlib.pyplot")
    sns = require_dependency("seaborn")

    palette = build_palette(hue_order)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=labels,
        hue_order=hue_order,
        palette=palette,
        s=args.point_size,
        linewidth=0,
        alpha=0.78,
    )
    plt.title("Phoneme-level Stylecode t-SNE")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(title="Phoneme", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.6)
    plt.tight_layout()
    plt.savefig(output_dir / "phoneme_stylecode_tsne.png", dpi=args.dpi)
    plt.savefig(output_dir / "phoneme_stylecode_tsne.pdf")
    plt.close()


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_points_csv(path: Path, coords: np.ndarray, points: list[Point]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["x", "y", "label", "raw_label", "id", "position", "step", "speaker", "token_id", "duration", "vq_index"],
        )
        writer.writeheader()
        for coord, point in zip(coords, points):
            writer.writerow(
                {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "label": point.label,
                    "raw_label": point.raw_label,
                    "id": point.sample_id,
                    "position": point.position,
                    "step": csv_value(point.step),
                    "speaker": csv_value(point.speaker),
                    "token_id": csv_value(point.token_id),
                    "duration": csv_value(point.duration),
                    "vq_index": csv_value(point.code_index),
                }
            )


def write_label_counts(path: Path, label_counts: Counter) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count"])
        writer.writeheader()
        for label, count in label_counts.most_common():
            writer.writerow({"label": label, "count": int(count)})


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    metrics = require_dependency("scikit-learn", "sklearn.metrics")

    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    try:
        return float(metrics.silhouette_score(features, labels))
    except ValueError:
        return None


def compute_metrics(features: np.ndarray, coords: np.ndarray, labels: list[str], seed: int) -> dict[str, Any]:
    neighbors = require_dependency("scikit-learn", "sklearn.neighbors")

    label_array = np.asarray(labels, dtype=object)
    label_counts = Counter(labels)
    majority_baseline = max(label_counts.values()) / len(labels)
    rng = np.random.default_rng(seed)
    metric_limit = min(5000, len(labels))
    if len(labels) > metric_limit:
        indices = rng.choice(len(labels), size=metric_limit, replace=False)
        metric_features = features[indices]
        metric_coords = coords[indices]
        metric_labels = label_array[indices]
    else:
        metric_features = features
        metric_coords = coords
        metric_labels = label_array

    nn = neighbors.NearestNeighbors(n_neighbors=2).fit(metric_features)
    nearest = nn.kneighbors(metric_features, return_distance=False)[:, 1]
    knn_same_label = float(np.mean(metric_labels[nearest] == metric_labels))

    return {
        "num_metric_points": int(len(metric_labels)),
        "majority_label_baseline": float(majority_baseline),
        "knn_same_label_accuracy_original_space": knn_same_label,
        "silhouette_original_space": safe_silhouette(metric_features, metric_labels),
        "silhouette_tsne_space": safe_silhouette(metric_coords, metric_labels),
    }


def compute_codebook_metrics(points: list[Point]) -> dict[str, Any] | None:
    metrics = require_dependency("scikit-learn", "sklearn.metrics")

    pairs = [(parse_code_index(point.code_index), point.label) for point in points]
    pairs = [(code, label) for code, label in pairs if code is not None]
    if not pairs:
        return None

    codes = [code for code, _ in pairs]
    labels = [label for _, label in pairs]
    code_counts = Counter(codes)
    label_counts = Counter(labels)
    total = len(pairs)
    probabilities = np.asarray([count / total for count in code_counts.values()], dtype=np.float64)
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) > 0 else 0.0
    perplexity = float(np.exp(entropy))
    inferred_codebook_size = max(codes) + 1
    configured_sizes = [parse_code_index(point.codebook_size) for point in points]
    configured_sizes = [size for size in configured_sizes if size is not None and size > 0]
    codebook_size = max(configured_sizes) if configured_sizes else inferred_codebook_size

    code_label_counts: dict[int, Counter] = defaultdict(Counter)
    label_code_counts: dict[str, Counter] = defaultdict(Counter)
    for code, label in pairs:
        code_label_counts[code][label] += 1
        label_code_counts[label][code] += 1

    code_purity = sum(max(counter.values()) for counter in code_label_counts.values()) / total
    label_to_code_accuracy = sum(max(counter.values()) for counter in label_code_counts.values()) / total
    code_to_label_accuracy = code_purity
    nmi = float(metrics.normalized_mutual_info_score(labels, codes))
    ami = float(metrics.adjusted_mutual_info_score(labels, codes))

    return {
        "num_code_points": int(total),
        "codebook_size": int(codebook_size),
        "used_codes": int(len(code_counts)),
        "unused_codes": int(max(codebook_size - len(code_counts), 0)),
        "code_usage_entropy": entropy,
        "code_usage_perplexity": perplexity,
        "code_usage_fraction": float(len(code_counts) / max(codebook_size, 1)),
        "code_phoneme_nmi": nmi,
        "code_phoneme_ami": ami,
        "code_predicts_phoneme_accuracy": float(code_to_label_accuracy),
        "phoneme_predicts_code_accuracy": float(label_to_code_accuracy),
        "code_phoneme_purity": float(code_purity),
        "majority_code_baseline": float(max(code_counts.values()) / total),
        "majority_label_baseline_for_code_points": float(max(label_counts.values()) / total),
        "top_codes": [
            {"code": int(code), "count": int(count)}
            for code, count in code_counts.most_common(20)
        ],
    }


def write_code_usage_csv(path: Path, points: list[Point]) -> None:
    code_counts = Counter()
    for point in points:
        code = parse_code_index(point.code_index)
        if code is not None:
            code_counts[code] += 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "count"])
        writer.writeheader()
        for code, count in code_counts.most_common():
            writer.writerow({"code": int(code), "count": int(count)})


def write_code_phoneme_csv(path: Path, points: list[Point]) -> None:
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for point in points:
        code = parse_code_index(point.code_index)
        if code is not None:
            counts[(code, point.label)] += 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "label", "count"])
        writer.writeheader()
        for (code, label), count in sorted(counts.items()):
            writer.writerow({"code": int(code), "label": label, "count": int(count)})


def plot_code_usage(points: list[Point], output_dir: Path, args: argparse.Namespace) -> bool:
    code_counts = Counter()
    for point in points:
        code = parse_code_index(point.code_index)
        if code is not None:
            code_counts[code] += 1
    if not code_counts:
        return False
    plt = require_dependency("matplotlib", "matplotlib.pyplot")
    codes = [code for code, _ in code_counts.most_common()]
    counts = [code_counts[code] for code in codes]
    plt.figure(figsize=(12, 5))
    plt.bar([str(code) for code in codes], counts)
    plt.title("VQ Code Usage")
    plt.xlabel("Code index")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / "vq_code_usage.png", dpi=args.dpi)
    plt.close()
    return True


def plot_code_phoneme_heatmap(points: list[Point], output_dir: Path, args: argparse.Namespace) -> bool:
    sns = require_dependency("seaborn")
    plt = require_dependency("matplotlib", "matplotlib.pyplot")
    counts: dict[tuple[int, str], int] = defaultdict(int)
    labels = []
    codes = []
    for point in points:
        code = parse_code_index(point.code_index)
        if code is None:
            continue
        counts[(code, point.label)] += 1
        labels.append(point.label)
        codes.append(code)
    if not counts:
        return False
    labels = [label for label, _ in Counter(labels).most_common(args.top_k)]
    codes = [code for code, _ in Counter(codes).most_common(min(50, len(set(codes))))]
    matrix = np.zeros((len(labels), len(codes)), dtype=np.float32)
    for row, label in enumerate(labels):
        for col, code in enumerate(codes):
            matrix[row, col] = counts.get((code, label), 0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, np.maximum(row_sums, 1.0))
    plt.figure(figsize=(max(10, len(codes) * 0.35), max(5, len(labels) * 0.35)))
    sns.heatmap(matrix, xticklabels=codes, yticklabels=labels, cmap="viridis")
    plt.title("P(code | phoneme)")
    plt.xlabel("Code index")
    plt.ylabel("Phoneme label")
    plt.tight_layout()
    plt.savefig(output_dir / "vq_phoneme_code_heatmap.png", dpi=args.dpi)
    plt.close()
    return True


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_paths(paths: list[str]) -> list[Path]:
    resolved = [Path(path) for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing stylecode JSONL file(s): " + ", ".join(missing))
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = validate_paths(args.stylecode_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_map = load_token_map(args.token_id_to_label_json)
    excluded = {
        normalize_label(label, args.keep_stress) for label in split_csv(args.exclude_labels)
    }

    label_counts, dim_counts, first_pass_stats = count_labels(paths, args, token_map, excluded)
    if not label_counts:
        raise RuntimeError("No valid phoneme positions found in the input JSONL file(s).")
    if len(dim_counts) != 1:
        raise RuntimeError(f"Expected one bottleneck dimension, got {dict(dim_counts)}. Run files with different dimensions separately.")
    expected_dim = next(iter(dim_counts.keys()))

    selected_labels, missing_explicit_labels = choose_labels(label_counts, args, excluded)
    if len(selected_labels) < 2:
        raise RuntimeError(
            "Need at least two selected labels. Lower --min_count, increase --top_k, or pass --labels. "
            f"Top counts: {label_counts.most_common(20)}"
        )

    grouped, other_points, second_pass_stats = collect_candidate_points(
        paths, args, token_map, excluded, selected_labels, expected_dim
    )
    points, sampled_counts = sample_points(grouped, other_points, selected_labels, args)
    labels = [point.label for point in points]
    if len(points) < 3:
        raise RuntimeError("Fewer than 3 sampled points remained after filtering and sampling.")
    if len(set(labels)) < 2:
        raise RuntimeError("Fewer than 2 sampled labels remained after filtering and sampling.")

    features = np.stack([point.vector for point in points], axis=0).astype(np.float32)
    normalized_features = preprocess_features(features, args.normalize)
    coords, reduction_info = reduce_tsne(normalized_features, args)
    hue_order = [label for label in selected_labels if sampled_counts.get(label, 0) > 0]
    if sampled_counts.get("other", 0) > 0:
        hue_order.append("other")

    plot_points(coords, labels, hue_order, output_dir, args)
    write_points_csv(output_dir / "phoneme_stylecode_points.csv", coords, points)
    write_label_counts(output_dir / "phoneme_label_counts.csv", label_counts)
    metrics = compute_metrics(normalized_features, coords, labels, args.seed)
    codebook_metrics = compute_codebook_metrics(points)
    if codebook_metrics is not None:
        metrics["codebook"] = codebook_metrics
        write_code_usage_csv(output_dir / "vq_code_usage.csv", points)
        write_code_phoneme_csv(output_dir / "vq_code_phoneme_counts.csv", points)
        plot_code_usage(points, output_dir, args)
        plot_code_phoneme_heatmap(points, output_dir, args)
    write_json(output_dir / "phoneme_stylecode_metrics.json", metrics)

    summary = {
        "input_paths": [str(path) for path in paths],
        "output_dir": str(output_dir),
        "bottleneck_dim": int(expected_dim),
        "selected_labels": selected_labels,
        "missing_explicit_labels": missing_explicit_labels,
        "selected_label_counts": {label: int(label_counts[label]) for label in selected_labels},
        "sampled_label_counts": {label: int(count) for label, count in sampled_counts.items()},
        "num_plotted_points": int(len(points)),
        "normalization": args.normalize,
        "reduction": reduction_info,
        "first_pass_stats": {key: int(value) for key, value in first_pass_stats.items()},
        "second_pass_stats": {key: int(value) for key, value in second_pass_stats.items()},
        "metrics": metrics,
        "parameters": {
            "top_k": args.top_k,
            "labels": args.labels,
            "min_count": args.min_count,
            "max_points": args.max_points,
            "max_per_label": args.max_per_label,
            "include_other": args.include_other,
            "max_other": args.max_other,
            "include_silence": args.include_silence,
            "exclude_labels": args.exclude_labels,
            "keep_stress": args.keep_stress,
            "pca_dim": args.pca_dim,
            "perplexity": args.perplexity,
            "seed": args.seed,
        },
        "outputs": {
            "png": str(output_dir / "phoneme_stylecode_tsne.png"),
            "pdf": str(output_dir / "phoneme_stylecode_tsne.pdf"),
            "points_csv": str(output_dir / "phoneme_stylecode_points.csv"),
            "label_counts_csv": str(output_dir / "phoneme_label_counts.csv"),
            "summary_json": str(output_dir / "phoneme_stylecode_summary.json"),
            "metrics_json": str(output_dir / "phoneme_stylecode_metrics.json"),
            "vq_code_usage_csv": str(output_dir / "vq_code_usage.csv"),
            "vq_code_phoneme_counts_csv": str(output_dir / "vq_code_phoneme_counts.csv"),
            "vq_code_usage_png": str(output_dir / "vq_code_usage.png"),
            "vq_phoneme_code_heatmap_png": str(output_dir / "vq_phoneme_code_heatmap.png"),
        },
    }
    write_json(output_dir / "phoneme_stylecode_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "eval_results" / "phoneme_stylecode"
    parser = argparse.ArgumentParser("Visualize phoneme-level FastSpeech2 stylecodes with t-SNE")
    parser.add_argument("--stylecode_jsonl", "--jsonl", nargs="+", required=True, help="One or more FastSpeech2 stylecode JSONL files.")
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--min_count", type=int, default=20)
    parser.add_argument("--max_points", type=int, default=8000)
    parser.add_argument("--max_per_label", type=int, default=600)
    parser.add_argument("--include_other", action="store_true")
    parser.add_argument("--max_other", type=int, default=1000)
    parser.add_argument("--include_silence", action="store_true")
    parser.add_argument("--exclude_labels", type=str, default="")
    parser.add_argument("--keep_stress", action="store_true")
    parser.add_argument("--token_id_to_label_json", type=str, default=None)
    parser.add_argument("--normalize", type=str, default="standard", choices=["standard", "l2", "none"])
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_size", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--fail_on_mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
