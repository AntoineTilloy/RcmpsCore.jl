"""Plot the fixed-precision D64 test; may also be run while it is in progress."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).resolve().parent/'d64_50'
with (p/'trajectory.csv').open() as stream:
    rows=[r for r in csv.DictReader(stream) if r.get('precond_reg')]
if not rows:
    raise SystemExit('No accepted iterates recorded yet.')
it=np.array([int(r['iteration']) for r in rows])
minutes=np.array([float(r['seconds'])/60 for r in rows])
energy=np.array([float(r['energy']) for r in rows])
gradient=np.array([float(r['gradient_norm']) for r in rows])
complete=it[-1]==50
fig,axes=plt.subplots(1,3,figsize=(11.5,3.7),layout='constrained')
color='#176b9c'
axes[0].plot(it,energy,color=color,lw=1.5)
axes[0].set(xlabel='Accepted LBFGS iterations',ylabel='Energy density',title=f'Final recorded E = {energy[-1]:.12f}')
axes[1].plot(minutes,energy,color=color,lw=1.5)
axes[1].set(xlabel='Optimization time (minutes)',ylabel='Energy density',title='Energy versus wall time')
axes[2].semilogy(it,gradient,color=color,lw=1.5)
axes[2].set(xlabel='Accepted LBFGS iterations',ylabel='Metric gradient norm',title=f'Final recorded norm = {gradient[-1]:.3e}')
for ax in axes:
    ax.grid(alpha=.2)
status='50 iterations completed; not a convergence claim' if complete else f'{it[-1]} iterations completed; in progress'
fig.suptitle(f'D = 64, phi4, g = 2 · {status}\nFixed fine precision · LBFGS history 100 · six compute threads · BLAS = 1',fontsize=11)
fig.savefig(p/'trajectory.png',dpi=180)
fig.savefig(p/'trajectory.pdf')
plt.close(fig)
