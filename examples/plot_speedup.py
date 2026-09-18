"""Trace le speedup pour les 3 versions."""

import pandas as pd
import matplotlib.pyplot as plt


configs = [
    ("Naive Dense", "speedup_naive.csv"),
    ("Sparse CSR",  "speedup_sparse.csv"),
    ("Mean-Field",  "speedup_meanfield.csv"),
]

colors = {"cython": "tab:orange", "c": "tab:red", "cpp": "tab:green"}
markers = {"cython": "s", "c": "o", "cpp": "D"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (title, csv_file) in zip(axes, configs):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        ax.set_title(f"{title}\n(CSV manquant)")
        continue

    for backend in ["cython", "c", "cpp"]:
        sub = df[df["backend"] == backend]
        if sub.empty:
            continue
        ax.plot(
            sub["N"], sub["speedup"],
            marker=markers[backend], color=colors[backend],
            linewidth=2, markersize=8, label=backend,
        )

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.4, label="1× (baseline)")
    ax.axhline(y=8.0, color="gray", linestyle=":", alpha=0.4, label="8× (max théorique)")
    ax.set_title(title)
    ax.set_xlabel("N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Speedup (temps 1T / temps 8T)")
fig.suptitle("Speedup OpenMP (8 threads vs 1 thread)", fontsize=14)

plt.tight_layout()
plt.savefig("benchmark_speedup_all.png", dpi=150)
plt.show()
print("✅ Sauvegardé : benchmark_speedup_all.png")
