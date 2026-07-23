"""
Plot a training_loss.csv log (written by utils.csv_logger.AveragedCSVLogger)
as a dual-axis loss / learning-rate figure, styled for a NeurIPS main-paper
figure.

CSV columns produced by the logger: epoch, step_start, step_end, num_steps,
avg_loss, lr. `step_end` resets to 0 at the start of every epoch, so this
script rebuilds a monotonic x-axis across epochs before plotting.

Usage:
    python plot.py path/to/training_loss.csv
    python plot.py path/to/training_loss.csv --config configs/qwen_reasondistill_final.yaml
    python plot.py path/to/training_loss.csv --config run.yaml --save --out figures/loss_lr

The figure is only written to disk when --save is passed; otherwise it is
just displayed (plt.show()).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

NEURIPS_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # embed as real text, not paths, in the PDF
    "ps.fonttype": 42,
}


def load_run(csv_path):
    """Load the CSV and add a monotonic `global_step` that stitches epochs together."""
    df = pd.read_csv(csv_path)
    epoch_len = df.groupby("epoch")["step_end"].max().sort_index()
    offset = epoch_len.cumsum().shift(1, fill_value=0)
    df["global_step"] = df["step_end"] + df["epoch"].map(offset)
    return df.sort_values("global_step").reset_index(drop=True)


def schedule_markers(config_path, data_max_step):
    """
    Recompute where warmup ends and (if used) where each cosine warm-restart
    happens, in the same step units as the CSV. Mirrors the logic in
    training/base_trainer.py so the markers line up with what the scheduler
    actually did; `data_max_step` (from the CSV itself) stands in for the
    planned total_opt_steps since that requires reloading the dataset.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())["training"]
    accum_steps = cfg["accum_steps"]

    total_opt_steps = max(1, data_max_step // accum_steps)
    warmup_opt_steps = max(1, round(total_opt_steps * cfg.get("warmup_ratio", 0.0)))
    markers = {"warmup_end": warmup_opt_steps * accum_steps, "restarts": []}

    if cfg.get("scheduler_type", "onecycle") == "cosine_restarts":
        period = cfg.get("restart_interval_steps") or max(1, (total_opt_steps - warmup_opt_steps) // 2)
        restart_mult = cfg.get("restart_mult", 1) or 1
        cursor = warmup_opt_steps
        while True:
            cursor += period
            if cursor >= total_opt_steps:
                break
            markers["restarts"].append(cursor * accum_steps)
            period *= restart_mult

    return markers


def steepest_drop_window(df, window=5):
    """Find the span of `window` consecutive logged rows with the single largest loss drop."""
    loss = df["avg_loss"].to_numpy()
    drops = loss[:-window] - loss[window:]
    i = int(np.argmax(drops))
    return df["global_step"].iloc[i], df["global_step"].iloc[i + window]


def make_plot(df, markers=None, title=None, wide=False):
    plt.rcParams.update(NEURIPS_RC)
    fig, ax_loss = plt.subplots(figsize=(7.2, 4.4) if wide else (5.2, 4.0))

    (loss_line,) = ax_loss.plot(df["global_step"], df["avg_loss"], color="#1a1a2e", zorder=3)
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Training loss")
    ax_loss.spines["top"].set_visible(False)
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)

    ax_lr = ax_loss.twinx()
    (lr_line,) = ax_lr.plot(df["global_step"], df["lr"], color="#c1440e", linestyle="--",
                             linewidth=1.2, alpha=0.85, zorder=2)
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_yscale("log")
    ax_lr.spines["top"].set_visible(False)

    # --- highlight interesting regions -----------------------------------
    # Every legend entry below is built explicitly (never from ax.get_lines())
    # so that axvline/axvspan helper artists never leak stray auto-generated
    # labels ("_child3", ...) into the legend.
    legend_handles = [
        Line2D([], [], color="#1a1a2e", label="Loss"),
        Line2D([], [], color="#c1440e", linestyle="--", label="LR"),
    ]

    # Steepest loss drop: always available, purely data-driven.
    lo, hi = steepest_drop_window(df)
    ax_loss.axvspan(lo, hi, color="#f6c453", alpha=0.3, zorder=1)
    legend_handles.append(Patch(facecolor="#f6c453", alpha=0.3, label="Steepest loss drop"))

    # Scheduler markers: only if a config was supplied.
    if markers is not None:
        ax_loss.axvline(markers["warmup_end"], color="#555555", linestyle=":", linewidth=1.0)
        for step in markers["restarts"]:
            ax_loss.axvline(step, color="#555555", linestyle=":", linewidth=1.0)
        label = "Warmup end / LR restart" if markers["restarts"] else "Warmup end"
        legend_handles.append(Line2D([], [], color="#555555", linestyle=":", label=label))

    if title:
        ax_loss.set_title(title, pad=8)

    # Legend goes below the axes so it never overlaps the curves, regardless
    # of their shape.
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to a training_loss.csv log")
    parser.add_argument("--config", default=None, help="YAML config used for the run (adds warmup/restart markers)")
    parser.add_argument("--save", action="store_true", help="Save the figure; otherwise it is only displayed")
    parser.add_argument("--out", default=None, help="Output path prefix when --save is passed (no extension)")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--wide", action="store_true", help="Double-column width instead of single-column")
    args = parser.parse_args()

    df = load_run(args.csv_path)
    markers = schedule_markers(args.config, df["global_step"].max()) if args.config else None
    fig = make_plot(df, markers=markers, title=args.title, wide=args.wide)

    if args.save:
        out = Path(args.out) if args.out else Path(args.csv_path).with_suffix("").with_name(
            Path(args.csv_path).stem + "_plot")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    else:
        plt.show()


if __name__ == "__main__":
    main()