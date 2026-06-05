from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from visualize_phoneme_stylecodes import run as run_single


def split_ints(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc


def split_floats(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one float value")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value}") from exc


def format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def build_single_args(args: argparse.Namespace, seed: int, perplexity: float, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stylecode_jsonl=args.stylecode_jsonl,
        output_dir=str(output_dir),
        top_k=args.top_k,
        labels=args.labels,
        min_count=args.min_count,
        max_points=args.max_points,
        max_per_label=args.max_per_label,
        include_other=args.include_other,
        max_other=args.max_other,
        include_silence=args.include_silence,
        exclude_labels=args.exclude_labels,
        keep_stress=args.keep_stress,
        token_id_to_label_json=args.token_id_to_label_json,
        normalize=args.normalize,
        pca_dim=args.pca_dim,
        perplexity=perplexity,
        seed=seed,
        point_size=args.point_size,
        dpi=args.dpi,
        fail_on_mismatch=args.fail_on_mismatch,
    )


def copy_figures(summary: dict[str, Any], figure_dir: Path, run_name: str) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    outputs = summary.get("outputs", {})
    figure_specs = [
        ("png", f"phoneme_stylecode_tsne_{run_name}.png"),
        ("pdf", f"phoneme_stylecode_tsne_{run_name}.pdf"),
        ("vq_code_usage_png", f"vq_code_usage_{run_name}.png"),
        ("vq_phoneme_code_heatmap_png", f"vq_phoneme_code_heatmap_{run_name}.png"),
    ]
    for key, filename in figure_specs:
        source_value = outputs.get(key)
        if not source_value:
            continue
        source = Path(source_value)
        if not source.exists():
            continue
        target = figure_dir / filename
        shutil.copy2(source, target)
        copied[key] = str(target)
    return copied


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "status",
        "seed",
        "perplexity",
        "output_dir",
        "figure_png",
        "figure_pdf",
        "vq_code_usage_png",
        "vq_phoneme_code_heatmap_png",
        "num_plotted_points",
        "selected_labels",
        "knn_same_label_accuracy_original_space",
        "codebook_used_codes",
        "codebook_usage_perplexity",
        "code_phoneme_nmi",
        "code_phoneme_purity",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def write_gallery_index(path: Path, rows: list[dict[str, Any]], figure_dir: Path) -> None:
    successful = [row for row in rows if row.get("status") == "ok" and row.get("figure_png")]
    html = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>Phoneme Stylecode t-SNE Batch</title>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 24px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 20px; }",
        "    .card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; }",
        "    img { width: 100%; height: auto; }",
        "    code { font-size: 13px; }",
        "  </style>",
        "</head>",
        "<body>",
        "<h1>Phoneme Stylecode t-SNE Batch</h1>",
        "<div class=\"grid\">",
    ]
    for row in successful:
        image_keys = [
            ("figure_png", "t-SNE"),
            ("vq_code_usage_png", "VQ usage"),
            ("vq_phoneme_code_heatmap_png", "P(code | phoneme)"),
        ]
        html.extend(
            [
                "  <div class=\"card\">",
                f"    <h2>{row['run_name']}</h2>",
                f"    <p><code>seed={row['seed']}, perplexity={row['perplexity']}</code></p>",
                f"    <p><code>labels={row.get('selected_labels', '')}</code></p>",
            ]
        )
        for key, title in image_keys:
            image_path = row.get(key)
            if not image_path:
                continue
            rel_png = Path(str(image_path)).relative_to(figure_dir)
            html.extend(
                [
                    f"    <h3>{title}</h3>",
                    f"    <a href=\"{rel_png}\"><img src=\"{rel_png}\" alt=\"{row['run_name']} {title}\"></a>",
                ]
            )
        html.append("  </div>")
    html.extend(["</div>", "</body>", "</html>"])
    path.write_text("\n".join(html) + "\n", encoding="utf-8")


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    runs_dir = output_root / "runs"
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for seed in args.seeds:
        for perplexity in args.perplexities:
            run_name = f"seed{seed}_perp{format_number(perplexity)}"
            run_output_dir = runs_dir / run_name
            single_args = build_single_args(args, seed, perplexity, run_output_dir)
            row: dict[str, Any] = {
                "run_name": run_name,
                "seed": seed,
                "perplexity": perplexity,
                "output_dir": str(run_output_dir),
            }
            try:
                summary = run_single(single_args)
                copied = copy_figures(summary, figure_dir, run_name)
                metrics = summary.get("metrics", {})
                codebook_metrics = metrics.get("codebook", {}) if isinstance(metrics, dict) else {}
                row.update(
                    {
                        "status": "ok",
                        "figure_png": copied.get("png", ""),
                        "figure_pdf": copied.get("pdf", ""),
                        "vq_code_usage_png": copied.get("vq_code_usage_png", ""),
                        "vq_phoneme_code_heatmap_png": copied.get("vq_phoneme_code_heatmap_png", ""),
                        "num_plotted_points": summary.get("num_plotted_points", ""),
                        "selected_labels": ",".join(summary.get("selected_labels", [])),
                        "knn_same_label_accuracy_original_space": metrics.get("knn_same_label_accuracy_original_space", "") if isinstance(metrics, dict) else "",
                        "codebook_used_codes": codebook_metrics.get("used_codes", ""),
                        "codebook_usage_perplexity": codebook_metrics.get("code_usage_perplexity", ""),
                        "code_phoneme_nmi": codebook_metrics.get("code_phoneme_nmi", ""),
                        "code_phoneme_purity": codebook_metrics.get("code_phoneme_purity", ""),
                    }
                )
            except Exception as exc:
                row.update({"status": "failed", "error": str(exc)})
                failures.append(row)
                if not args.continue_on_error:
                    rows.append(row)
                    write_manifest_csv(output_root / "batch_manifest.csv", rows)
                    raise
            rows.append(row)

    write_manifest_csv(output_root / "batch_manifest.csv", rows)
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    write_json(
        output_root / "batch_summary.json",
        {
            "input_paths": args.stylecode_jsonl,
            "output_root": str(output_root),
            "runs_dir": str(runs_dir),
            "figure_dir": str(figure_dir),
            "seeds": args.seeds,
            "perplexities": args.perplexities,
            "num_runs": len(rows),
            "num_successful": len(successful_rows),
            "num_failed": len(failures),
            "metric_means": {
                "knn_same_label_accuracy_original_space": mean_numeric(successful_rows, "knn_same_label_accuracy_original_space"),
                "codebook_used_codes": mean_numeric(successful_rows, "codebook_used_codes"),
                "codebook_usage_perplexity": mean_numeric(successful_rows, "codebook_usage_perplexity"),
                "code_phoneme_nmi": mean_numeric(successful_rows, "code_phoneme_nmi"),
                "code_phoneme_purity": mean_numeric(successful_rows, "code_phoneme_purity"),
            },
            "failures": failures,
            "manifest_csv": str(output_root / "batch_manifest.csv"),
            "gallery_index": str(figure_dir / "index.html"),
        },
    )
    write_gallery_index(figure_dir / "index.html", rows, figure_dir)
    return {
        "output_root": str(output_root),
        "runs_dir": str(runs_dir),
        "figure_dir": str(figure_dir),
        "manifest_csv": str(output_root / "batch_manifest.csv"),
        "summary_json": str(output_root / "batch_summary.json"),
        "gallery_index": str(figure_dir / "index.html"),
        "num_runs": len(rows),
        "num_successful": sum(1 for row in rows if row.get("status") == "ok"),
        "num_failed": len(failures),
    }


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "eval_results" / "phoneme_stylecode_batch"
    parser = argparse.ArgumentParser("Run phoneme-level stylecode t-SNE over multiple seeds and perplexities")
    parser.add_argument("--stylecode_jsonl", "--jsonl", nargs="+", required=True, help="One or more FastSpeech2 stylecode JSONL files.")
    parser.add_argument("--output_root", type=str, default=str(default_output))
    parser.add_argument("--figure_dir", type=str, default="", help="Directory that collects all PNG/PDF figures. Defaults to output_root/figures.")
    parser.add_argument("--seeds", type=split_ints, default=split_ints("1,42,123"), help="Comma-separated seeds, e.g. 1,42,123.")
    parser.add_argument("--perplexities", type=split_floats, default=split_floats("15,30,50"), help="Comma-separated perplexities, e.g. 15,30,50.")
    parser.add_argument("--continue_on_error", action="store_true")

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
    parser.add_argument("--point_size", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--fail_on_mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_batch(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
