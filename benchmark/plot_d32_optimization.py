"""Plot accepted D32 optimization iterates; safe to run on a partial CSV."""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'd32_optimization'
summary = {}
status_path = root / 'status.json'
run_status = json.loads(status_path.read_text()) if status_path.exists() else {}
stopped = run_status.get('status') == 'stopped_by_user'
if (root / 'summary.csv').exists():
    with (root / 'summary.csv').open() as stream:
        summary = {int(float(r['coupling'])): r for r in csv.DictReader(stream)}
couplings = (2,) if stopped and not (root / 'g1_trajectory.csv').exists() else (2, 1)
fig, axes = plt.subplots(len(couplings), 3, figsize=(12, 3.6*len(couplings)), layout='constrained', squeeze=False)
colors = {2: '#176b9c', 1: '#b35b26'}
for row, coupling in enumerate(couplings):
    source = root / f'g{coupling}_trajectory.csv'
    records = []
    if source.exists():
        with source.open() as stream:
            records = [r for r in csv.DictReader(stream) if r.get('gradient_norm')]
    if not records:
        for ax in axes[row]:
            ax.text(.5, .5, f'g = {coupling}: pending', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
        continue
    iteration = np.array([int(r['iteration']) for r in records])
    minutes = np.array([float(r['seconds']) / 60 for r in records])
    energy = np.array([float(r['energy']) for r in records])
    gradient = np.array([float(r['gradient_norm']) for r in records])
    verified = summary.get(coupling, {}).get('converged') == 'true'
    reference = float(summary[coupling]['energy']) if verified else energy[-1]
    color = colors[coupling]
    axes[row, 0].plot(iteration, energy, color=color, lw=1.4)
    energy_error = abs(energy-reference)
    energy_error[energy_error <= np.finfo(float).eps*max(1., abs(reference))] = np.nan
    axes[row, 1].semilogy(minutes, energy_error, color=color, lw=1.4)
    axes[row, 2].semilogy(minutes, gradient, color=color, lw=1.4)
    axes[row, 2].axhline(3e-6, color='0.4', lw=.8, ls=':', label='Fine stopping target')
    axes[row, 2].legend(frameon=False, fontsize=8)
    axes[row, 0].set(xlabel='Accepted LBFGS iterations', ylabel='Energy density')
    axes[row, 1].set(xlabel='Optimization time (minutes)', ylabel='|E − E_final|' if verified else '|E − last logged E|')
    axes[row, 2].set(xlabel='Optimization time (minutes)', ylabel='Metric gradient norm')
    status = 'verified converged' if verified else ('stopped, unconverged' if stopped else 'in progress')
    axes[row, 0].set_title(f'g = {coupling} · {status}\nE = {reference:.12f}', fontsize=10)
    axes[row, 1].set_title('Energy convergence', fontsize=10)
    axes[row, 2].set_title('Gradient convergence', fontsize=10)
    transitions = [i for i in range(1, len(records)) if records[i]['phase'] != records[i-1]['phase']]
    for i in transitions:
        for col, ax in enumerate(axes[row]):
            ax.axvline(iteration[i] if col == 0 else minutes[i], color='0.5', ls='--', lw=.8)
    for ax in axes[row]:
        ax.grid(alpha=.2, which='major')
heading = ('D = 32, phi4, g = 2: stopped by request; g = 1 was not started' if stopped else
           'D = 32, phi4: g = 2 from a random state, then g = 1 from the optimized g = 2 state')
fig.suptitle(heading+'\nSix compute threads · BLAS = 1 · Krylov kinetics · dashed lines mark precision refinement', fontsize=11)
root.mkdir(parents=True, exist_ok=True)
fig.savefig(root / 'optimization_trajectories.png', dpi=180)
fig.savefig(root / 'optimization_trajectories.pdf')
plt.close(fig)
