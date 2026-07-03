import argparse
import json
import os
import tempfile


def load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to draw the summary figure; install the FastSpeech2 requirements first"
        ) from exc
    return plt


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


def format_checkpoint(restore_step):
    if restore_step % 1000 == 0:
        return "{}k".format(restore_step // 1000)
    return str(restore_step)


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = data.get("rows")
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError("summary_json must contain a list or a dict with a 'rows' list")
    return rows


def collect_plot_data(rows, cfgs, sample_steps, metric):
    selected_cfgs = [float(cfg) for cfg in cfgs]
    selected_steps = [int(step) for step in sample_steps]
    step_index = {step: index for index, step in enumerate(selected_steps)}
    values = {}
    warnings = []

    restore_steps = sorted({int(row["restore_step"]) for row in rows if "restore_step" in row})
    for cfg in selected_cfgs:
        values[cfg] = {restore_step: [None] * len(selected_steps) for restore_step in restore_steps}

    for row in rows:
        try:
            cfg = float(row["guidance_scale"])
            sample_step = int(row["sample_steps"])
            restore_step = int(row["restore_step"])
        except KeyError as exc:
            warnings.append("skipping row missing {}".format(exc.args[0]))
            continue
        if cfg not in values or sample_step not in step_index:
            continue
        if metric not in row:
            warnings.append("skipping row missing {} for cfg={}, checkpoint={}, sample_steps={}".format(metric, cfg, restore_step, sample_step))
            continue
        index = step_index[sample_step]
        if values[cfg][restore_step][index] is not None:
            warnings.append("duplicate value for cfg={}, checkpoint={}, sample_steps={}; using the last one".format(cfg, restore_step, sample_step))
        values[cfg][restore_step][index] = float(row[metric])

    for cfg in selected_cfgs:
        for restore_step in restore_steps:
            missing_steps = [str(step) for step, value in zip(selected_steps, values[cfg][restore_step]) if value is None]
            if missing_steps and len(missing_steps) < len(selected_steps):
                warnings.append("cfg={}, checkpoint={} missing sample_steps: {}".format(cfg, restore_step, ",".join(missing_steps)))
    return values, warnings


def checkpoint_colors(plt, restore_steps):
    if not restore_steps:
        return {}
    if len(restore_steps) == 1:
        return {restore_steps[0]: plt.cm.Blues(0.75)}
    return {
        restore_step: plt.cm.Blues(0.35 + 0.55 * index / float(len(restore_steps) - 1))
        for index, restore_step in enumerate(restore_steps)
    }


def plot_summary(rows, output_path, cfgs, sample_steps, metric="score", title="Flow-mel ablation summary", dpi=200):
    plt = load_pyplot()
    plot_data, warnings = collect_plot_data(rows, cfgs, sample_steps, metric)
    restore_steps = sorted({restore_step for cfg_data in plot_data.values() for restore_step in cfg_data})
    if not restore_steps:
        raise ValueError("No restore_step values found in summary rows")

    colors = checkpoint_colors(plt, restore_steps)
    x_positions = list(range(len(sample_steps)))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.8 * len(cfgs), 4.2), sharey=True)
    if len(cfgs) == 1:
        axes = [axes]

    handles_by_step = {}
    for axis, cfg in zip(axes, cfgs):
        cfg = float(cfg)
        cfg_data = plot_data.get(cfg, {})
        for restore_step in restore_steps:
            series = cfg_data.get(restore_step, [None] * len(sample_steps))
            points = [(x, y) for x, y in zip(x_positions, series) if y is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            line, = axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                color=colors[restore_step],
                label=format_checkpoint(restore_step),
            )
            handles_by_step.setdefault(restore_step, line)
        axis.set_title("CFG={}".format(cfg))
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(step) for step in sample_steps])
        axis.set_xlabel("Sample steps")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(metric)
    fig.suptitle(title)

    handles = [handles_by_step[restore_step] for restore_step in restore_steps if restore_step in handles_by_step]
    labels = [format_checkpoint(restore_step) for restore_step in restore_steps if restore_step in handles_by_step]
    if handles:
        fig.legend(handles, labels, title="Checkpoint", loc="center right", bbox_to_anchor=(0.995, 0.5))
        fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.92))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return warnings


def run(args):
    rows = load_summary(args.summary_json)
    warnings = plot_summary(
        rows,
        args.output,
        args.cfgs,
        args.sample_steps,
        metric=args.metric,
        title=args.title,
        dpi=args.dpi,
    )
    for warning in warnings:
        print("warning: {}".format(warning))
    print("Saved plot to {}".format(args.output))


def self_test():
    rows = [
        {"restore_step": 500000, "guidance_scale": 1.0, "sample_steps": 16, "score": 0.30},
        {"restore_step": 500000, "guidance_scale": 1.0, "sample_steps": 32, "score": 0.25},
        {"restore_step": 700000, "guidance_scale": 1.0, "sample_steps": 16, "score": 0.22},
        {"restore_step": 700000, "guidance_scale": 1.5, "sample_steps": 64, "score": 0.20},
    ]
    plot_data, warnings = collect_plot_data(rows, [1.0, 1.5], [1, 16, 32, 64], "score")
    assert plot_data[1.0][500000] == [None, 0.30, 0.25, None]
    assert plot_data[1.0][700000] == [None, 0.22, None, None]
    assert plot_data[1.5][700000] == [None, None, None, 0.20]
    assert any("missing" in warning for warning in warnings)

    try:
        load_pyplot()
    except RuntimeError:
        print("plot smoke skipped: matplotlib is not available")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = os.path.join(temp_dir, "summary.json")
            output_path = os.path.join(temp_dir, "flow_mel_summary.png")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"rows": rows}, f)
            loaded_rows = load_summary(summary_path)
            plot_warnings = plot_summary(loaded_rows, output_path, [1.0, 1.5], [1, 16, 32, 64])
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            assert plot_warnings
    print("plot_flow_mel_summary self_test ok")


def parse_args():
    parser = argparse.ArgumentParser("Plot flow-mel summary ablation curves")
    parser.add_argument("--self_test", action="store_true", help="Run helper self test and exit")
    parser.add_argument("--summary_json", type=str, default="", help="Path to evaluate_flow_mel.py summary.json")
    parser.add_argument("--output", type=str, default="flow_mel_summary.png", help="Output figure path (.png, .pdf, .svg, ...)")
    parser.add_argument("--metric", type=str, default="score", help="Numeric row field to plot on the Y axis")
    parser.add_argument("--cfgs", type=split_floats, default=split_floats("1.0,1.5,2.0"), help="Comma-separated CFG values to draw as columns")
    parser.add_argument("--sample_steps", type=split_ints, default=split_ints("1,16,32,64"), help="Comma-separated sample steps for the X axis")
    parser.add_argument("--title", type=str, default="Flow-mel ablation summary")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.summary_json:
            raise SystemExit("Missing required argument: --summary_json")
        run(args)
