"""Trace les 3 benchmarks (Naive, Sparse, Mean-Field) à 8 threads."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

configs = [
    ("Naive Dense", "execution_times_naive_8threads.csv", "tab:red", "o"),
    ("Sparse CSR",  "execution_times_sparse_8threads.csv", "tab:green", "D"),
    ("Mean-Field",  "execution_times_meanfield_8threads.csv", "tab:orange", "s"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (title, csv_file, color, marker) in zip(axes, configs):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        ax.set_title(f"{title}\n(CSV manquant)")
        continue

    # Tracer uniquement les backends natifs (cython, c, cpp)
    # + python pour référence
    for backend in ["python", "cython", "c", "cpp"]:
        sub = df[df["backend"] == backend]
        if sub.empty:
            continue
        ax.loglog(
            sub["N"], sub["mean_s"],
            marker=marker if backend == "c" else "o",
            label=backend,
            linewidth=2,
            markersize=8,
        )

    ax.set_title(f"{title} (8 threads)")
    ax.set_xlabel("N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Temps (s)")
fig.suptitle("Benchmark à 8 threads : Naive vs Sparse vs Mean-Field", fontsize=14)

plt.tight_layout()
plt.savefig("benchmark_8threads_comparison.png", dpi=150)
plt.show()
print("✅ Sauvegardé : benchmark_8threads_comparison.png")