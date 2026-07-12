import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes for readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# Read the CSV file
df = pd.read_csv('training_loss.csv')

# Use step_start as x-axis (each row is an aggregated block)
x = df['step_start']

# Create figure and primary axis for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss
ax1.plot(x, df['avg_loss'], label='Training Loss', color='blue', linewidth=2)
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.6)

# Secondary axis for learning rate
ax2 = ax1.twinx()
ax2.plot(x, df['lr'], label='Learning Rate', color='red', linewidth=2)
ax2.set_ylabel('Learning Rate', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_yscale('log')  # log scale for LR to better show decay

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Title
ax1.set_title('Distillation Training: Loss and Learning Rate')

# ------------------------------------------------------------
# Highlight important features
# ------------------------------------------------------------

# 1. Warm-up phase: LR increases until it peaks (find max LR)
peak_lr_idx = df['lr'].idxmax()
peak_step = df.loc[peak_lr_idx, 'step_start']
# Shade the warm-up region (from start to peak)
ax1.axvspan(x.iloc[0], peak_step, alpha=0.15, color='gray', label='Warm-up')
# Annotate peak
ax1.annotate('Warm-up ends\nLR peaks',
             xy=(peak_step, df.loc[peak_lr_idx, 'avg_loss']),
             xytext=(peak_step + 500, df.loc[peak_lr_idx, 'avg_loss'] + 0.2),
             arrowprops=dict(arrowstyle='->', color='darkorange'),
             fontsize=11, color='darkorange')

# 2. Steepest loss drop: find where loss drops fastest (max negative gradient)
loss = df['avg_loss'].values
grad = np.gradient(loss)
steepest_idx = np.argmin(grad)  # most negative
steepest_step = x.iloc[steepest_idx]
steepest_loss = loss[steepest_idx]
ax1.annotate('Steepest drop',
             xy=(steepest_step, steepest_loss),
             xytext=(steepest_step + 300, steepest_loss + 0.3),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=11, color='green')

# 3. Loss plateau: horizontal line at final loss
final_loss = loss[-1]
ax1.axhline(y=final_loss, color='blue', linestyle=':', alpha=0.7, linewidth=1.5,
            label=f'Final loss = {final_loss:.4f}')

# 4. Learning rate decay: mark where LR drops below 1e-5 (if applicable)
# (LR is already decaying after peak; we can annotate start of significant decay)
# We can just add a vertical line at the start of decay (after warm-up)
ax2.axvline(x=peak_step, color='red', linestyle=':', alpha=0.5, linewidth=1)

# Adjust layout and save
plt.tight_layout()
plt.savefig('distillation_training_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('distillation_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as 'distillation_training_curves.pdf' and 'distillation_training_curves.png'")