#!/usr/bin/env python3
"""Plot smoothed loss traces from a training log."""

import argparse
import csv
import json
import math
from pathlib import Path


BOOKKEEPING_COLUMNS = {
    "epoch",
    "step",
    "global_step",
    "step_start",
    "step_end",
    "num_steps",
    "lr",
    "learning_rate",
}

PAPER_COLORS = {
    "smoothed": "#1f77b4",
    "raw": "#b8c2cc",
    "grid": "#d8dee9",
    "text": "#1f2933",
    "spine": "#9aa5b1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot smoothed loss components for a distillation scheme."
    )
    parser.add_argument("pos_log_path", nargs="?", type=Path, help="Path to a CSV or JSONL loss log.")
    parser.add_argument("pos_scheme", nargs="?", help="Scheme name to use in plot titles/output names.")
    parser.add_argument("--log_path", type=Path, default=None, help="Path to a CSV or JSONL loss log.")
    parser.add_argument("--scheme", default=None, help="Scheme name to use in plot titles/output names.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log_dir>/<scheme>_losses.png.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="EMA smoothing factor in [0, 1). Larger values are smoother.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window in addition to saving the image.",
    )
    args = parser.parse_args()
    args.log_path = args.log_path or args.pos_log_path
    args.scheme = args.scheme or args.pos_scheme
    if args.log_path is None or args.scheme is None:
        parser.error("provide a log path and scheme, either positionally or with --log_path/--scheme")
    return args


def pretty_name(name):
    replacements = {
        "avg": "Average",
        "ce": "CE",
        "kd": "KD",
        "kl": "KL",
        "mse": "MSE",
        "lr": "LR",
    }
    words = str(name).replace("-", "_").split("_")
    return " ".join(replacements.get(word.lower(), word.capitalize()) for word in words)


def configure_plot_style():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PAPER_COLORS["spine"],
        "axes.labelcolor": PAPER_COLORS["text"],
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.facecolor": "white",
        "xtick.color": PAPER_COLORS["text"],
        "xtick.labelsize": 9,
        "ytick.color": PAPER_COLORS["text"],
        "ytick.labelsize": 9,
    })
    return plt


def read_rows(log_path):
    suffix = log_path.suffix.lower()
    if suffix == ".jsonl":
        with log_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    with log_path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def choose_x_column(rows):
    for name in ("global_step", "step_end", "step", "epoch"):
        if name in rows[0]:
            return name
    return None


def choose_loss_columns(rows):
    numeric_columns = []
    for name in rows[0]:
        values = [as_float(row.get(name)) for row in rows]
        if any(value is not None for value in values):
            numeric_columns.append(name)

    loss_columns = [
        name
        for name in numeric_columns
        if "loss" in name.lower() and name.lower() not in BOOKKEEPING_COLUMNS
    ]
    if loss_columns:
        return loss_columns

    return [
        name
        for name in numeric_columns
        if name.lower() not in BOOKKEEPING_COLUMNS and "step" not in name.lower()
    ]


def ema(values, smooth):
    if not 0 <= smooth < 1:
        raise ValueError("--smooth must be in [0, 1).")

    smoothed = []
    last = None
    for value in values:
        if value is None:
            smoothed.append(None)
            continue
        last = value if last is None else smooth * last + (1 - smooth) * value
        smoothed.append(last)
    return smoothed


def plot_losses(log_path, scheme, output_path=None, smooth=0.9, show=False):
    plt = configure_plot_style()

    rows = read_rows(log_path)
    if not rows:
        raise SystemExit(f"No rows found in {log_path}.")

    x_column = choose_x_column(rows)
    x_values = (
        [as_float(row.get(x_column)) for row in rows]
        if x_column is not None
        else list(range(1, len(rows) + 1))
    )
    if any(value is None for value in x_values):
        x_values = list(range(1, len(rows) + 1))
        x_label = "log row"
    else:
        x_label = x_column or "log row"

    loss_columns = choose_loss_columns(rows)
    if not loss_columns:
        raise SystemExit(f"No numeric loss columns found in {log_path}.")

    ncols = 1 if len(loss_columns) <= 2 else 2
    nrows = math.ceil(len(loss_columns) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 2.65 * nrows + 0.45),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for idx, (axis, column) in enumerate(zip(axes.ravel(), loss_columns)):
        values = [as_float(row.get(column)) for row in rows]
        smooth_values = ema(values, smooth)
        clean_points = [
            (x, raw, smooth_y)
            for x, raw, smooth_y in zip(x_values, values, smooth_values)
            if raw is not None and smooth_y is not None
        ]
        clean_x = [point[0] for point in clean_points]
        clean_y = [point[1] for point in clean_points]
        clean_smooth_y = [point[2] for point in clean_points]

        axis.plot(clean_x, clean_y, color=PAPER_COLORS["raw"], alpha=0.28, linewidth=0.8)
        axis.plot(clean_x, clean_smooth_y, color=PAPER_COLORS["smoothed"], linewidth=2.25)
        axis.set_title(pretty_name(column), loc="left", pad=8)
        axis.set_ylabel("Loss")
        axis.grid(True, axis="y", color=PAPER_COLORS["grid"], alpha=0.75, linewidth=0.8)
        axis.grid(False, axis="x")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        axis.tick_params(axis="both", length=3, width=0.8)

        final_value = clean_smooth_y[-1] if clean_smooth_y else None
        if final_value is not None:
            axis.annotate(
                f"{final_value:.3g}",
                xy=(clean_x[-1], final_value),
                xytext=(6, 0),
                textcoords="offset points",
                color=PAPER_COLORS["smoothed"],
                fontsize=9,
                fontweight="bold",
                va="center",
            )

        if idx == 0:
            axis.text(
                0.995,
                0.97,
                f"EMA {smooth:g}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#52606d",
                fontsize=8.5,
            )

    for axis in axes.ravel()[len(loss_columns):]:
        axis.set_visible(False)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            axis = axes[row_idx, col_idx]
            if not axis.get_visible():
                continue
            visible_below = any(
                axes[other_row, col_idx].get_visible()
                for other_row in range(row_idx + 1, nrows)
            )
            if not visible_below:
                axis.set_xlabel(pretty_name(x_label))
                axis.tick_params(axis="x", labelbottom=True)

    fig.suptitle(
        f"{pretty_name(scheme)} Losses",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=PAPER_COLORS["text"],
    )

    output_path = output_path or log_path.with_name(f"{scheme}_losses.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main():
    args = parse_args()
    output_path = plot_losses(
        args.log_path,
        args.scheme,
        output_path=args.output,
        smooth=args.smooth,
        show=args.show,
    )
    print(output_path)


if __name__ == "__main__":
    main()
