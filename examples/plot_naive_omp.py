"""Trace Naive Dense multi-threads (3, 5, 8) — 3 subplots côte à côte."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Chargement
# ============================================================

df = pd.read_csv("execution_times_naive_omp_threads.csv")

print(df.head())
print("\nBackends :", df["backend"].unique())
print("Threads :", df["n_threads"].unique())
print("N :", df["N"].unique())


# ============================================================
# Paramètres
# ============================================================

backends_order = ["c", "cpp", "cython"]
threads = sorted(df["n_threads"].unique())

colors = {3: "tab:blue", 5: "tab:orange", 8: "tab:red"}
markers = {3: "o", 5: "s", 8: "^"}
linestyles = {3: "-", 5: "--", 8: ":"}


# ============================================================
# FONCTION : droites de référence
# ============================================================

def add_reference_lines(ax, N_min, y_ref):
    N_ref = np.array([N_min, N_min * 40])

    factor_N = y_ref / N_min
    ax.loglog(N_ref, factor_N * N_ref,
              "k--", alpha=0.4, linewidth=1.5, label=r"$O(N)$")

    factor_N2 = y_ref / (N_min ** 2)
    ax.loglog(N_ref, factor_N2 * N_ref ** 2,
              "k:", alpha=0.4, linewidth=1.5, label=r"$O(N^2)$")


# ============================================================
# FIGURE : 3 subplots côte à côte
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(11, 5),       # ⭐ même taille que ton cas 1 courbe (6×5 par subplot)
    sharey=True,
)

for ax, backend in zip(axes, backends_order):
    data_backend = df[df["backend"] == backend]

    for n_threads in threads:
        sub = data_backend[
            data_backend["n_threads"] == n_threads
        ].sort_values("N")

        if sub.empty:
            continue

        ax.loglog(
            sub["N"], sub["mean_s"],
            marker=markers[n_threads],
            color=colors[n_threads],
            linestyle=linestyles[n_threads],
            linewidth=2,
            markersize=9,
            label=f"{n_threads} threads",
        )

    # Références
    N_min = data_backend["N"].min()
    y_ref = data_backend["mean_s"].min()
    add_reference_lines(ax, N_min, y_ref)

    ax.set_title(backend.upper(), fontsize=13)
    ax.set_xlabel("N", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")


axes[0].set_ylabel("Temps (s)", fontsize=12)

fig.suptitle("Naive Dense : effet du nombre de threads (3, 5, 8)",
             fontsize=15, y=1.00)

plt.tight_layout()
plt.savefig("exec_time_ompnaive_.png", dpi=200, bbox_inches="tight")
plt.show()
print("✅ exec_time_ompnaive_.png")