"""Plot the Julia finite-difference benchmark (requires numpy and matplotlib)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]
data = np.genfromtxt(root/'benchmark/boundary.csv', delimiter=',', names=True)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, constrained_layout=True)
for ax, gap in zip(axes, sorted(set(data['gap']), reverse=True)):
    rows = data[data['gap']==gap]
    ax.semilogy(rows['window'], rows['bulk_error'], 'o-', color='#bc4b3d', label='Bulk only')
    ax.semilogy(rows['window'], rows['corrected_error'], 'o-', color='#1b7290', label='Boundary corrected')
    ax.set_title(f'Transfer gap = {gap:.4g}')
    ax.set_xlabel('Physical half-window L')
    ax.set_xticks([4,8,16,32])
    ax.grid(alpha=.2, which='major')
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Absolute directional-derivative error')
axes[0].legend(frameon=False, loc='best')
fig.suptitle('Initial-state differentiation removes the slowly decaying gradient error')
fig.savefig(root/'docs/boundary.png', dpi=180)
fig.savefig(root/'docs/boundary.svg')
