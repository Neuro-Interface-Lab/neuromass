"""Trace la mémoire GPU pour les 4 cas sur un même graphe."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




CSV_NAIVE     = "08_memoire_gpu_par_N.csv"
CSV_SPARSE    = "08_memoire_gpu_sparse_par_N.csv"
CSV_LINEAR    = "gpu_memory_sparse_linear.csv"
CSV_QUADRATIC = "gpu_memory_sparse_quadratic.csv"       




df_naive   = pd.read_csv(CSV_NAIVE).sort_values("N")
df_sparse  = pd.read_csv(CSV_SPARSE).sort_values("N")
df_linear  = pd.read_csv(CSV_LINEAR).sort_values("N")
df_quad    = pd.read_csv(CSV_QUADRATIC).sort_values("N")

# Filtrer les valeurs aberrantes
df_linear = df_linear[df_linear["mem_gpu_MB"] > 0]
df_quad   = df_quad[df_quad["mem_gpu_MB"] > 0]




fig, ax = plt.subplots(figsize=(7, 6))

#  Naive dense
ax.plot(
    df_naive["N"], df_naive["total_GPU_memory_MB"],
    marker="s", linewidth=2, markersize=8,
    color="tab:blue",
    label="Naive dense (M = N²)",
)
for x, y in zip(df_naive["N"], df_naive["total_GPU_memory_MB"]):
    ax.annotate(f"{y:.1f}", (x, y),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="tab:blue")


#  d = 10 constant
ax.plot(
    df_sparse["N"], df_sparse["total_GPU_memory_MB"],
    marker="o", linewidth=2, markersize=8,
    color="tab:green",
    label="Sparse d=10 (M = 10N)",
)
for x, y in zip(df_sparse["N"], df_sparse["total_GPU_memory_MB"]):
    ax.annotate(f"{y:.1f}", (x, y),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="tab:green")


#  M = K·N²
ax.plot(
    df_linear["N"], df_linear["mem_gpu_MB"],
    marker="D", linewidth=2, markersize=8,
    color="tab:red",
    label="Sparse d ∝ N (M = K·N²)",
)
for x, y in zip(df_linear["N"], df_linear["mem_gpu_MB"]):
    ax.annotate(f"{y:.1f}", (x, y),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color="tab:red")


#  M = K₂·N³
ax.plot(
    df_quad["N"], df_quad["mem_gpu_MB"],
    marker="^", linewidth=2, markersize=9,
    color="tab:purple",
    label="Sparse d ∝ N² (M = K₂·N³)",
)
for x, y in zip(df_quad["N"], df_quad["mem_gpu_MB"]):
    ax.annotate(f"{y:.1f}", (x, y),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color="tab:purple")



# Références asymptotiques

N_min = min(
    df_naive["N"].min(),
    df_sparse["N"].min(),
    df_linear["N"].min(),
    df_quad["N"].min(),
)

y_ref = min(
    df_naive["total_GPU_memory_MB"].min(),
    df_sparse["total_GPU_memory_MB"].min(),
    df_linear["mem_gpu_MB"].min(),
    df_quad["mem_gpu_MB"].min(),
)

N_ref = np.array([N_min, N_min * 200])

# O(N) — ancrée à (N_min, y_ref)
factor_N = y_ref / N_min
ax.loglog(N_ref, factor_N * N_ref,
          "k--", alpha=0.4, linewidth=1.5, label=r"$O(N)$")

# O(N²) — ancrée à (N_min, y_ref)
factor_N2 = y_ref / (N_min ** 2)
ax.loglog(N_ref, factor_N2 * N_ref ** 2,
          "k:", alpha=0.4, linewidth=1.5, label=r"$O(N^2)$")

# O(N³) — ancrée à (N_min, y_ref)
factor_N3 = y_ref / (N_min ** 3)
ax.loglog(N_ref, factor_N3 * N_ref ** 3,
          "k-.", alpha=0.4, linewidth=1.5, label=r"$O(N^3)$")





ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Nombre de nœuds N (log)", fontsize=12)
ax.set_ylabel("Mémoire GPU (Mo, log)", fontsize=12)
ax.set_title("Mémoire GPU en fonction de N — Comparaison des 4 cas",
             fontsize=13, pad=15)

ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("memoire_GPU_comparaison_4cas.png", dpi=150, bbox_inches="tight")
plt.show()
print(" memoire_GPU_comparaison_4cas.png")