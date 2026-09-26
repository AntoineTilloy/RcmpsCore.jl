"""Render measured thread scaling and full-optimization results."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

p = Path(__file__).resolve().parent / "threading"
rows = []
for path in p.glob("evaluations_t*_b1.csv"):
    with path.open() as stream:
        rows.extend(csv.DictReader(stream))

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained", sharey=True)
for d, ax in zip((4, 8, 16), axes):
    values = {(int(r["julia_threads"]), r["mode"]): float(r["median_seconds"])
              for r in rows if int(r["D"]) == d and r["state"] == "converged"}
    for mode, label, color in (("terms", "Observable terms", "#176b9c"),
                                ("trajectories", "Forward/backward", "#b35b26"),
                                ("both", "Both levels", "#427745")):
        nt = (1, 2, 3, 6)
        ax.plot(nt, [values[t, "serial"] / values[t, mode] for t in nt],
                "o-", label=label, color=color)
    ax.axhline(1, color="0.5", linewidth=.8, linestyle="--")
    ax.set(title=f"D = {d}", xlabel="Julia compute threads", xticks=(1, 2, 3, 6))
    ax.grid(alpha=.2)
axes[0].set_ylabel("Speedup over serial evaluation")
axes[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Converged states at g = 2: fixed accuracy, BLAS = 1\n"
             "Median of three warmed evaluations", fontsize=11)
fig.savefig(p / "thread_scaling.png", dpi=180)
fig.savefig(p / "thread_scaling.pdf")

with (p / "optimizations.csv").open() as stream:
    final = {(int(r["D"]), r["kinetic_backend"], r["mode"]): r
             for r in csv.DictReader(stream)}
lines = ["## Measured results", "", "| D | Serial ODE | Threaded ODE | Threaded + resolvent kinetics |",
         "|---|---:|---:|---:|"]
for d in (4, 8, 16):
    parts = []
    for backend, mode in (("ode", "serial"), ("ode", "both"), ("resolvent", "both")):
        r = final[d, backend, mode]
        if r["converged"] != "true":
            raise ValueError(f"Unconverged result: {r}")
        parts.append(f"{float(r['seconds']):.1f} s / {r['iterations']} iterations")
    lines.append(f"| {d} | " + " | ".join(parts) + " |")
lines += ["", "Times are complete optimization costs, excluding warmup, saved-state I/O, and",
          "independent final validation. Each process has six compute threads and one",
          "BLAS/GC thread; the serial mode performs no concurrent observable work.",
          "The evaluator and optimizer are warmed, although first-use callback specialization",
          "can still affect the shortest D4 measurement.", "",
          "| D | Serial ODE energy | Threaded ODE energy | Resolvent-backend state, ODE-verified energy |",
          "|---|---:|---:|---:|"]
for d in (4, 8, 16):
    parts = [f"{float(final[d, b, m]['energy']):.12f}"
             for b, m in (("ode", "serial"), ("ode", "both"), ("resolvent", "both"))]
    lines.append(f"| {d} | " + " | ".join(parts) + " |")
lines += ["", "![Thread scaling](thread_scaling.png)", "",
          "[Full optimization data](optimizations.csv) · [Component profiles](components.csv)", ""]
readme = p / "README.md"
text = readme.read_text()
head, protocol = text.split("## Protocol", 1)
head = head.split("## Measured results", 1)[0]
readme.write_text(head + "\n".join(lines) + "\n## Protocol" + protocol)
