"""Plot a snapshot of one Z2 scan trajectory, including its unresolved gradient."""
import csv
import json
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

case=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
out.mkdir(parents=True,exist_ok=True)
with (case/"trajectory.csv").open() as f:
    records=[r for r in csv.DictReader(f) if r.get("stationary_residual")]
assert records, "No accepted iterations available"
status=tomllib.loads((case/"status.toml").read_text())
iteration=np.array([int(r["iteration"]) for r in records])
minutes=np.array([float(r["seconds"])/60 for r in records])
energy=np.array([float(r["energy"]) for r in records])
gradient=np.array([float(r["gradient_norm"]) for r in records])
assert np.all(np.diff(iteration)>0) and np.all(np.diff(minutes)>=0)
assert np.all(np.isfinite(energy)) and np.all(gradient>0)
target=float(status["target_gradient_norm"])
tail=iteration>=max(iteration[0],iteration[-1]-100)
color="#176b9c"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.titlesize":11,"savefig.facecolor":"white"})
fig,axes=plt.subplots(1,3,figsize=(13.5,4.6))
fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.72,wspace=.37)
axes[0].plot(iteration,energy,color=color,lw=1.6)
axes[0].set(title="Energy: full trajectory",xlabel="Accepted LBFGS iterations",ylabel="Energy density")
axes[1].plot(minutes[tail],energy[tail],color=color,lw=1.6)
axes[1].plot(minutes[-1],energy[-1],"o",color=color,ms=4)
axes[1].set(title="Energy: last 100 iterations",xlabel="Optimization time (minutes)",ylabel="Energy density")
axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[2].semilogy(minutes,gradient,color=color,lw=1.4,label="Metric gradient norm")
axes[2].axhline(target,color="#b34d28",ls="--",lw=1.3,label=f"Target: {target:.0e}")
axes[2].plot(minutes[-1],gradient[-1],"o",color=color,ms=4)
axes[2].set(title="Gradient convergence",xlabel="Optimization time (minutes)",ylabel="Metric gradient norm")
axes[2].legend(frameon=False,fontsize=9,loc="upper right")
for ax in axes:
    ax.grid(alpha=.2,which="major")
    ax.tick_params(axis="both",labelsize=9)
state_label="verified converged" if status.get("converged") else "not yet converged"
fig.suptitle(f"Z₂-symmetric φ⁴ state · D = {status['D']} · g = {status['coupling']:.3f}",y=.97,fontsize=15)
fig.text(.5,.865,f"{iteration[-1]} accepted iterations · {minutes[-1]/60:.2f} hours · {state_label}",ha="center",fontsize=11)
fig.text(.5,.805,f"Last E = {energy[-1]:.12f}     |     Last gradient = {gradient[-1]:.3e} ({gradient[-1]/target:.0f}× target)",ha="center",fontsize=10)
fig.text(.065,.045,"Fresh random start · fixed precision · LBFGS history 100 · Atlas: 2 Julia threads, BLAS = 1",fontsize=9,color=".35")
stem=f"D{status['D']}_g{status['coupling']:.4f}_iteration{iteration[-1]}"
fig.savefig(out/f"{stem}.png",dpi=180)
fig.savefig(out/f"{stem}.pdf")
plt.close(fig)
with (out/f"{stem}.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]))
    writer.writeheader();writer.writerows(records)
(out/f"{stem}.json").write_text(json.dumps({"source":str(case),"status_at_snapshot":status,
    "last_plotted_iteration":int(iteration[-1]),"last_plotted_energy":float(energy[-1]),
    "last_plotted_gradient":float(gradient[-1]),"optimization_minutes":float(minutes[-1])},indent=2)+"\n")
print(out/f"{stem}.png")
print(out/f"{stem}.pdf")
