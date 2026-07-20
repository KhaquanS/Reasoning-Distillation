import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# 1. Load and prepare data
# ----------------------------------------------------------------------
normal = pd.read_csv("https://huggingface.co/Khaquan/qwen-khaquanS-distillations/resolve/main/reasondistill_logs/normal_run/training_loss.csv")
continue_df = pd.read_csv("https://huggingface.co/Khaquan/qwen-khaquanS-distillations/resolve/main/reasondistill_logs/continue_run/training_loss.csv")

# Align step indices: continue run starts after normal run ends
normal_end_step = normal["step_start"].max()
continue_shifted = continue_df.copy()
continue_shifted["step_start"] += normal_end_step
continue_shifted["step_end"] += normal_end_step

# Combine
combined = pd.concat([
    normal.assign(run="Initial"),
    continue_shifted.assign(run="Continued")
], ignore_index=True)

# Mark the continuation point
split_step = normal_end_step

# ----------------------------------------------------------------------
# 2. Create the plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

fig, ax1 = plt.subplots(figsize=(9, 5.5))

# --- Loss (left axis) ---
color_loss = "#2c3e50"  # dark slate
ax1.set_xlabel("Optimization Step")
ax1.set_ylabel("Training Loss", color=color_loss)
ax1.tick_params(axis="y", labelcolor=color_loss)
ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)

# Plot loss for both runs
for run_name, group in combined.groupby("run"):
    ax1.plot(group["step_start"], group["avg_loss"],
             label=f"{run_name} Loss",
             color="#3498db" if run_name == "Initial" else "#e67e22",
             linewidth=1.8,
             alpha=0.9)

# --- Learning Rate (right axis, log scale) ---
ax2 = ax1.twinx()
color_lr = "#7f8c8d"  # gray
ax2.set_ylabel("Learning Rate", color=color_lr)
ax2.tick_params(axis="y", labelcolor=color_lr)
ax2.set_yscale("log")

for run_name, group in combined.groupby("run"):
    ax2.plot(group["step_start"], group["lr"],
             label=f"{run_name} LR",
             color="#95a5a6" if run_name == "Initial" else "#d35400",
             linewidth=1.2,
             linestyle="--",
             alpha=0.7)

# ----------------------------------------------------------------------
# 3. Highlight key events
# ----------------------------------------------------------------------

# 3a. Warm‑up region (initial run only)
warmup_end = normal[normal["lr"] == normal["lr"].max()]["step_start"].values[0]
ax1.axvspan(0, warmup_end, alpha=0.08, color="#2c3e50", label="Warm‑up")
ax1.annotate("Warm‑up ends", xy=(warmup_end, 1.1),
             xytext=(warmup_end + 200, 1.5),
             arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8),
             fontsize=9, color="#2c3e50")

# 3b. Steepest loss drop (initial run)
loss_vals = normal["avg_loss"].values
grad = np.gradient(loss_vals)
steepest_idx = np.argmin(grad)
steepest_step = normal.iloc[steepest_idx]["step_start"]
steepest_loss = loss_vals[steepest_idx]
ax1.annotate("Fastest drop", xy=(steepest_step, steepest_loss),
             xytext=(steepest_step + 300, steepest_loss - 0.3),
             arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8),
             fontsize=9, color="#27ae60")

# 3c. Continuation point (vertical dashed line)
ax1.axvline(x=split_step, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7)
ax1.annotate("Continue from\nfinal checkpoint",
             xy=(split_step, 0.75),
             xytext=(split_step + 400, 0.65),
             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8),
             fontsize=9, color="#e74c3c",
             ha="center")

# 3d. Final loss plateau (continued run)
final_loss = continue_df["avg_loss"].iloc[-1]
ax1.axhline(y=final_loss, color="#8e44ad", linestyle="-.", linewidth=1.2, alpha=0.6,
            label=f"Final loss = {final_loss:.3f}")

# ----------------------------------------------------------------------
# 4. Polish and save
# ----------------------------------------------------------------------
# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                    loc="upper right", frameon=False, ncol=2)

# Title and layout
ax1.set_title("ReasonDistill Training: Loss and Learning Rate", pad=15)
fig.tight_layout()

# Save
plt.savefig("distillation_combined_curves.pdf", bbox_inches="tight")
plt.savefig("distillation_combined_curves.png", bbox_inches="tight")
print("Plots saved as 'distillation_combined_curves.pdf' and '.png'")