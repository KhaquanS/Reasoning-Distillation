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
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
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


def calculate_lr_schedule(config_path, total_steps):
    """
    Manually calculate learning rate for every training step based on config.
    Returns arrays of step indices and corresponding LR values.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())["training"]
    
    # Extract parameters
    accum_steps = cfg["accum_steps"]
    total_opt_steps = total_steps // accum_steps
    
    lr_peak = cfg["lr"]
    min_lr = cfg["min_lr"]
    warmup_ratio = cfg.get("warmup_ratio", 0.05)
    scheduler_type = cfg.get("scheduler_type", "onecycle")
    
    warmup_opt_steps = max(1, round(total_opt_steps * warmup_ratio))
    
    # Initialize arrays for step-by-step LR values
    all_steps = np.arange(total_steps)
    lr_values = np.zeros(total_steps)
    
    # Warmup phase: linear increase from 0 to lr_peak
    warmup_steps = warmup_opt_steps * accum_steps
    warmup_indices = np.arange(warmup_steps)
    lr_values[warmup_indices] = lr_peak * (warmup_indices / warmup_steps)
    
    if scheduler_type == "cosine_restarts":
        # Cosine annealing with warm restarts (SGDR)
        restart_interval_opt = cfg.get("restart_interval_steps", 70)
        restart_mult = cfg.get("restart_mult", 1)
        
        # Start cosine decay after warmup
        cursor_opt = warmup_opt_steps
        period_opt = restart_interval_opt
        
        while cursor_opt < total_opt_steps:
            end_opt = min(cursor_opt + period_opt, total_opt_steps)
            steps_in_cycle = end_opt - cursor_opt
            steps_in_cycle_total = period_opt
            
            # Convert optimizer steps to training steps
            cursor = cursor_opt * accum_steps
            end = end_opt * accum_steps
            steps_in_cycle_train = end - cursor
            steps_in_cycle_total_train = steps_in_cycle_total * accum_steps
            
            # Cosine decay within this cycle
            cycle_indices = np.arange(cursor, end)
            progress = np.arange(steps_in_cycle_train) / steps_in_cycle_total_train
            # Cosine annealing: from lr_peak down to min_lr, then jump back to lr_peak at restart
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            lr_values[cycle_indices] = min_lr + (lr_peak - min_lr) * cosine_factor
            
            # Update for next cycle
            cursor_opt = end_opt
            period_opt *= restart_mult
            
    else:  # onecycle or default cosine decay
        # Simple cosine decay from lr_peak to min_lr after warmup
        decay_steps = total_steps - warmup_steps
        decay_indices = np.arange(warmup_steps, total_steps)
        progress = np.arange(decay_steps) / decay_steps
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        lr_values[decay_indices] = min_lr + (lr_peak - min_lr) * cosine_factor
    
    return all_steps, lr_values


def schedule_markers(config_path, data_max_step):
    """
    Recompute where warmup ends and where each cosine warm-restart happens,
    in the same step units as the CSV.
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
    
    # If there are fewer rows than the window, return the full range
    if len(loss) <= window:
        return df["global_step"].iloc[0], df["global_step"].iloc[-1]
    
    drops = loss[:-window] - loss[window:]
    i = int(np.argmax(drops))
    return df["global_step"].iloc[i], df["global_step"].iloc[i + window]


def make_plot(df, config_path=None, markers=None, title=None, wide=False):
    plt.rcParams.update(NEURIPS_RC)
    
    # Reduced figure size
    fig, ax_loss = plt.subplots(figsize=(4.0, 3.0) if wide else (3.5, 2.8))
    
    # Plot loss curve (from CSV)
    (loss_line,) = ax_loss.plot(df["global_step"], df["avg_loss"], color="#1a1a2e", zorder=3)
    ax_loss.set_xlabel("Training step", fontsize=8)
    ax_loss.set_ylabel("Training loss", fontsize=8)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.35)
    
    # Plot learning rate (manually calculated if config provided, else from CSV)
    ax_lr = ax_loss.twinx()
    
    if config_path:
        # Manually calculate LR for every step
        all_steps, lr_values = calculate_lr_schedule(config_path, df["global_step"].max())
        (lr_line,) = ax_lr.plot(all_steps, lr_values, color="#c1440e", linestyle="-", 
                                linewidth=0.8, alpha=0.7, zorder=2)
    else:
        # Fallback to CSV's lr column (only logs at checkpoint steps)
        (lr_line,) = ax_lr.plot(df["global_step"], df["lr"], color="#c1440e", linestyle="--", 
                                linewidth=1.0, alpha=0.85, zorder=2)
    
    ax_lr.set_ylabel("Learning rate", fontsize=8)
    ax_lr.set_yscale("log")
    ax_lr.spines["top"].set_visible(False)
    
    # --- highlight interesting regions -----------------------------------
    legend_handles = [
        Line2D([], [], color="#1a1a2e", label="Loss"),
        Line2D([], [], color="#c1440e", linestyle="-" if config_path else "--", label="LR"),
    ]
    
    # Steepest loss drop
    lo, hi = steepest_drop_window(df)
    ax_loss.axvspan(lo, hi, color="#f6c453", alpha=0.3, zorder=1)
    legend_handles.append(Patch(facecolor="#f6c453", alpha=0.3, label="Steepest loss drop"))
    
    # Scheduler markers
    if markers is not None:
        ax_loss.axvline(markers["warmup_end"], color="#555555", linestyle=":", linewidth=0.8)
        for step in markers["restarts"]:
            ax_loss.axvline(step, color="#555555", linestyle=":", linewidth=0.8)
        label = "Warmup end / LR restart" if markers["restarts"] else "Warmup end"
        legend_handles.append(Line2D([], [], color="#555555", linestyle=":", label=label))
    
    if title:
        ax_loss.set_title(title, pad=6, fontsize=9)
    
    # Legend below the axes
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        fontsize=7,
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
    fig = make_plot(df, config_path=args.config, markers=markers, title=args.title, wide=args.wide)
    
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