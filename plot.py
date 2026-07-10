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
df = pd.read_csv('sae_training_matrics.csv')

# Convert token counts to millions for a cleaner x-axis
x = df['tokens_start'] / 1e6  # in millions

# Create a 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# ------------------------------------------------------------
# Subplot 1: Total Loss and Reconstruction Loss
# ------------------------------------------------------------
ax1.plot(x, df['avg_total_loss'], label='Total Loss', color='blue', linewidth=2)
ax1.set_xlabel('Tokens seen (millions)')
ax1.set_ylabel('Total Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.6)

# Reconstruction loss on secondary y-axis (much smaller)
ax1b = ax1.twinx()
ax1b.plot(x, df['avg_reconstruction_loss'], label='Reconstruction Loss', color='red', linewidth=2)
ax1b.set_ylabel('Reconstruction Loss', color='red')
ax1b.tick_params(axis='y', labelcolor='red')

# Add legend (combine both axes)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax1.set_title('Loss Evolution')

# ------------------------------------------------------------
# Subplot 2: Sparsity Loss (log scale)
# ------------------------------------------------------------
ax2.plot(x, df['avg_sparsity_loss'], label='Sparsity Loss', color='green', linewidth=2)
ax2.set_xlabel('Tokens seen (millions)')
ax2.set_ylabel('Sparsity Loss')
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_title('Sparsity Loss (log scale)')

# Highlight where sparsity loss drops below 0.01 (roughly)
threshold = 0.01
idx = np.where(df['avg_sparsity_loss'] < threshold)[0]
if len(idx) > 0:
    first_idx = idx[0]
    x_ann = x.iloc[first_idx]
    y_ann = df['avg_sparsity_loss'].iloc[first_idx]
    ax2.axvline(x_ann, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.annotate(f'Sparsity loss < {threshold}',
                 xy=(x_ann, y_ann),
                 xytext=(x_ann + 5, y_ann * 10),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=11, color='darkgreen')

# ------------------------------------------------------------
# Subplot 3: Sparsity Metrics – L0 and L1 norms
# ------------------------------------------------------------
ax3.plot(x, df['avg_l0_norm'], label='L0 Norm (active features)', color='purple', linewidth=2)
ax3.set_xlabel('Tokens seen (millions)')
ax3.set_ylabel('L0 Norm', color='purple')
ax3.tick_params(axis='y', labelcolor='purple')
ax3.grid(True, linestyle='--', alpha=0.6)

ax3b = ax3.twinx()
ax3b.plot(x, df['avg_l1_norm'], label='L1 Norm', color='orange', linewidth=2)
ax3b.set_ylabel('L1 Norm', color='orange')
ax3b.tick_params(axis='y', labelcolor='orange')

lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax3b.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
ax3.set_title('Sparsity Norms')

# Annotate stabilization of L0 norm
stable_l0 = df['avg_l0_norm'].iloc[-1] * 1.05  # near final value
idx_stable = np.where(df['avg_l0_norm'] < stable_l0)[0]
if len(idx_stable) > 0:
    start_stable = idx_stable[0]
    x_stable = x.iloc[start_stable]
    ax3.axvline(x_stable, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.annotate('L0 stabilizes',
                 xy=(x_stable, df['avg_l0_norm'].iloc[start_stable]),
                 xytext=(x_stable + 2, df['avg_l0_norm'].iloc[start_stable] * 1.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=11, color='darkmagenta')

# ------------------------------------------------------------
# Subplot 4: Learning Rate Schedule
# ------------------------------------------------------------
ax4.plot(x, df['lr'], label='Learning Rate', color='black', linewidth=2)
ax4.set_xlabel('Tokens seen (millions)')
ax4.set_ylabel('Learning Rate')
ax4.set_yscale('log')
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.set_title('Learning Rate Decay')

# Highlight the start of decay (first drop from 2e-5)
initial_lr = df['lr'].iloc[0]
decay_start = np.where(df['lr'] < initial_lr)[0]
if len(decay_start) > 0:
    start_idx = decay_start[0]
    x_decay = x.iloc[start_idx]
    ax4.axvline(x_decay, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax4.annotate('LR begins to decay',
                 xy=(x_decay, df['lr'].iloc[start_idx]),
                 xytext=(x_decay + 2, df['lr'].iloc[start_idx] * 2),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=11)

# ------------------------------------------------------------
# Final touches
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig('sae_training_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sae_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as 'sae_training_curves.pdf' and 'sae_training_curves.png'")