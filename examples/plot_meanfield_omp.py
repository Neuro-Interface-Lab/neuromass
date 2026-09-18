"""Trace organisé des courbes Mean-Field multi-threads (3, 5, 8)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "execution_times_meanfield_omp_threads.csv"

df = pd.read_csv(CSV_FILE)

if "n_threads" not in df.columns:
    raise ValueError(f"Le CSV {CSV_FILE} doit contenir une colonne 'n_threads'.")

# EXCLURE python
backends = sorted([b for b in df["backend"].unique() if b != "python"])
thread_counts = sorted(df["n_threads"].unique())


# ============================================================
# STYLES PAR NOMBRE DE THREADS
# ============================================================

# Couleur par thread
colors = {
    3: "tab:blue",
    5: "tab:green",
    8: "tab:red",
}

# Marqueur par thread
markers = {
    3: "o",   # cercle
    5: "s",   # carré
    8: "^",   # triangle
}

# Style de ligne par thread
linestyles = {
    3: "-",    # plein
    5: "--",   # tirets
    8: ":",    # pointillés
}


# ============================================================
# FONCTION : ajouter les droites de référence
# ============================================================

def add_reference_lines(ax, N_min, y_ref):
    """Ajoute les droites O(N) et O(N²) ancrées à (N_min, y_ref)."""
    N_ref = np.array([N_min, N_min * 40])

    factor_N = y_ref / N_min
    ax.loglog(N_ref, factor_N * N_ref,
              "k--", alpha=0.5, linewidth=1.5, label=r"$O(N)$")

    factor_N2 = y_ref / (N_min ** 2)
    ax.loglog(N_ref, factor_N2 * N_ref ** 2,
              "k:", alpha=0.5, linewidth=1.5, label=r"$O(N^2)$")


# ============================================================
# FIGURE 1 : Tout sur une figure
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

for n_threads in thread_counts:
    for backend in backends:
        sub = df[(df["backend"] == backend) & (df["n_threads"] == n_threads)]
        if sub.empty:
            continue

        ax.loglog(
            sub["N"], sub["mean_s"],
            marker=markers[n_threads],
            color=colors[n_threads],
            linestyle=linestyles[n_threads],
            linewidth=2,
            markersize=9,
            label=f"{n_threads}T — {backend}",
        )

N_min = df[df["backend"] != "python"]["N"].min()
y_ref = df[df["backend"] != "python"]["mean_s"].min()
add_reference_lines(ax, N_min, y_ref)

ax.set_xlabel("Nombre d'oscillateurs (N)")
ax.set_ylabel("Temps d'exécution (s)")
ax.set_title("Mean-Field : Séquentiel vs OpenMP (3, 5, 8 threads)")
ax.legend(fontsize=8, ncol=2, loc="best")
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_meanfield_omp_threads_all.png", dpi=150)
plt.show()
print("✅ benchmark_meanfield_omp_threads_all.png")


# ============================================================
# FIGURE 2 : Un subplot par backend
# ============================================================

n_backends = len(backends)
fig, axes = plt.subplots(1, n_backends, figsize=(5 * n_backends, 5), sharey=True)
if n_backends == 1:
    axes = [axes]

for ax, backend in zip(axes, backends):
    sub_backend = df[df["backend"] == backend]

    for n_threads in thread_counts:
        sub = sub_backend[sub_backend["n_threads"] == n_threads]
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

    N_min = sub_backend["N"].min()
    y_ref = sub_backend["mean_s"].min()
    add_reference_lines(ax, N_min, y_ref)

    ax.set_title(f"{backend.upper()}")
    ax.set_xlabel("N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Temps (s)")
fig.suptitle("Mean-Field : effet du nombre de threads (3, 5, 8)", fontsize=14)

plt.tight_layout()
plt.savefig("benchmark_meanfield_omp_threads_perbackend.png", dpi=150)
plt.show()
print("✅ benchmark_meanfield_omp_threads_perbackend.png")


# ============================================================
# FIGURE 3 : Speedup par rapport à 3 threads
# ============================================================

fig, axes = plt.subplots(1, n_backends, figsize=(5 * n_backends, 5), sharey=True)
if n_backends == 1:
    axes = [axes]

for ax, backend in zip(axes, backends):
    pivot = df[df["backend"] == backend].pivot(
        index="N", columns="n_threads", values="mean_s"
    )
    if pivot.empty or 3 not in pivot.columns:
        ax.set_title(f"{backend.upper()} (pas de données 3T)")
        continue

    ref = pivot[3]
    for n_threads in thread_counts:
        if n_threads == 3 or n_threads not in pivot.columns:
            continue
        speedup = ref / pivot[n_threads]

        ax.plot(
            pivot.index, speedup,
            marker=markers[n_threads],
            color=colors[n_threads],
            linestyle=linestyles[n_threads],
            linewidth=2,
            markersize=9,
            label=f"{n_threads}T / 3T",
        )

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    ax.set_title(f"{backend.upper()}")
    ax.set_xlabel("N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Speedup (temps 3T / temps nT)")
fig.suptitle("Mean-Field : speedup par rapport à 3 threads", fontsize=14)

plt.tight_layout()
plt.savefig("benchmark_meanfield_omp_threads_speedup.png", dpi=150)
plt.show()
print("✅ benchmark_meanfield_omp_threads_speedup.png")


# ============================================================
# FIGURE 4 : Heatmap (sans python)
# ============================================================

N_max = df["N"].max()
sub = df[(df["N"] == N_max) & (df["backend"] != "python")]
pivot = sub.pivot(index="backend", columns="n_threads", values="mean_s")

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.4f}s", ha="center", va="center",
                    color="white", fontsize=10)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{c}T" for c in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("Nombre de threads")
ax.set_ylabel("Backend")
ax.set_title(f"Mean-Field : temps d'exécution à N = {N_max}")
plt.colorbar(im, ax=ax, label="Temps (s)")

plt.tight_layout()
plt.savefig("benchmark_meanfield_omp_heatmap.png", dpi=150)
plt.show()
print("✅ benchmark_meanfield_omp_heatmap.png")