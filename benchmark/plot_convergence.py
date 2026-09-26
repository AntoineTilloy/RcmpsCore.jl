"""Plot optimizer histories and independently checked final gradient norms."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent / "converged"

def read_history(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    complete = [r for r in rows if r.get("iteration") and r.get("optimizer_gradient_norm")]
    if len(complete) != len(rows):
        print(f"{path.name}: skipping {len(rows) - len(complete)} incomplete log rows")
    return complete

with (folder / "summary.csv").open() as stream:
    latest = {(int(r["D"]), r["implementation"]): r for r in csv.DictReader(stream)}

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True, layout="constrained")
for dimension, ax in zip((2, 4, 8, 16), axes.flat):
    for name, color in (("old", "#b35b26"), ("new", "#176b9c")):
        history = read_history(folder / f"D{dimension}_{name}.csv")
        xs = [int(r["iteration"]) for r in history]
        ys = [float(r["optimizer_gradient_norm"]) for r in history]
        stages = sorted(folder.glob(f"D{dimension}_{name}_polish_attempt*.csv"))
        stages += [folder / f"D{dimension}_{name}_polish.csv"]
        for polish in stages:
            if not polish.exists():
                continue
            offset = max(xs)
            extra = read_history(polish)
            xs += [offset + int(r["iteration"]) for r in extra]
            ys += [float(r["optimizer_gradient_norm"]) for r in extra]
        ax.semilogy(xs, ys, label=name.capitalize(), color=color, linewidth=1.5)
        final = latest[dimension, name]
        ax.scatter(int(final["iterations"]), float(final["corrected_gradient_norm"]),
                   marker="o", color=color, s=30, zorder=3)
    ax.axhline(1e-5, color="0.35", linestyle="--", linewidth=0.8)
    ax.set(title=f"D = {dimension}", xlabel="LBFGS iteration", ylim=(2e-7, 1e3))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
for ax in axes[:, 0]:
    ax.set_ylabel("Gradient norm")
fig.suptitle("Full optimization at g = 2: matched starting matrices\n"
             "Dots: corrected final gradients; dashed line: acceptance threshold", fontsize=12)
fig.savefig(folder / "convergence.png", dpi=180)
fig.savefig(folder / "convergence.pdf")
