"""Plot the completed Atlas short optimization benchmark."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "atlas_blas"
with (root / "summary.csv").open() as f:
    rows = list(csv.DictReader(f))
labels = []
seen = set()
for row in rows:
    nt = row["blas_threads"]
    labels.append(nt + ("\nrepeat" if nt in seen else ""))
    seen.add(nt)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, key, title in zip(axes,
        ("optimization_seconds", "median_fg_seconds"),
        ("Three LBFGS iterations", "Median energy/gradient evaluation")):
    values = [float(row[key]) for row in rows]
    colors = ["#168477" if value == min(values) else "#397bac" for value in values]
    ax.bar(range(len(rows)), values, color=colors)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("OpenBLAS threads")
    ax.set_ylabel("Seconds")
    ax.set_title(title)
    ax.set_ylim(0, max(values)*1.15)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle("Atlas · Threadripper 7960X · D = 64, g = 2\n"
             "Six Julia threads · fixed fine precision · identical starting state")
fig.savefig(root / "comparison.png", dpi=180)
fig.savefig(root / "comparison.pdf")
