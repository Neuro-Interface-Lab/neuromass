"""Trace les métriques de speedup, efficiency et loi d'Amdahl."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_FILE = "speedup_analysis.csv"   # ← changer selon ta version

df = pd.read_csv(CSV_FILE)

# Colonnes attendues :
# N, backend, t1, t8, speedup, efficiency, f_serial, parallel_fraction

P = 8   # nombre de threads

backends = sorted(df["backend"].unique())

colors = {
    "python": "tab:blue",
    "cython": "tab:orange",
    "c":      "tab:red",
    "cpp":    "tab:green",
}
markers = {
    "python": "v",
    "cython": "s",
    "c":      "o",
    "cpp":    "D",
}


# ============================================================
# FIGURE 1 : Speedup
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(
        sub["N"], sub["speedup"],
        marker=markers.get(backend, "o"),
        color=colors.get(backend, "gray"),
        linewidth=2, markersize=8,
        label=backend,
    )

ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="1× (baseline)")
ax.axhline(y=P, color="gray", linestyle=":", alpha=0.3, label=f"{P}× (max théorique)")

ax.set_xlabel("N")
ax.set_ylabel("Speedup  S = T1 / T8")
ax.set_title("Speedup OpenMP (8 threads vs 1 thread)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_speedup.png", dpi=150)
plt.show()
print("✅ plot_speedup.png")


# ============================================================
# FIGURE 2 : Efficiency
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(
        sub["N"], sub["efficiency"],
        marker=markers.get(backend, "o"),
        color=colors.get(backend, "gray"),
        linewidth=2, markersize=8,
        label=backend,
    )

ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="1.0 (parfait)")
ax.set_xlabel("N")
ax.set_ylabel("Efficiency  E = S / P")
ax.set_title(f"Efficacité OpenMP (P = {P} threads)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_efficiency.png", dpi=150)
plt.show()
print("✅ plot_efficiency.png")


# ============================================================
# FIGURE 3 : Fraction parallèle vs séquentielle
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(
        sub["N"], sub["parallel_fraction"],
        marker=markers.get(backend, "o"),
        color=colors.get(backend, "gray"),
        linestyle="-", linewidth=2, markersize=8,
        label=f"{backend} (parallèle)",
    )
    ax.plot(
        sub["N"], sub["f_serial"],
        marker="x",
        color=colors.get(backend, "gray"),
        linestyle="--", linewidth=1.5, markersize=6,
        alpha=0.6,
        label=f"{backend} (séquentiel)",
    )

ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.3)
ax.axhline(y=0.0, color="k", linestyle=":", alpha=0.3)
ax.set_xlabel("N")
ax.set_ylabel("Fraction")
ax.set_title("Fractions parallèle et séquentielle (estimation Amdahl)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_fractions.png", dpi=150)
plt.show()
print("✅ plot_fractions.png")


# ============================================================
# FIGURE 4 : Comparaison Speedup mesuré vs Amdahl théorique
# ============================================================

def amdahl_speedup(f_serial, p):
    """Loi d'Amdahl : S(p) = 1 / (f_serial + (1 - f_serial)/p)."""
    return 1.0 / (f_serial + (1.0 - f_serial) / p)


fig, ax = plt.subplots(figsize=(10, 6))

for backend in backends:
    sub = df[df["backend"] == backend]
    if sub.empty:
        continue

    # Speedup mesuré
    ax.plot(
        sub["N"], sub["speedup"],
        marker=markers.get(backend, "o"),
        color=colors.get(backend, "gray"),
        linewidth=2, markersize=8,
        label=f"{backend} (mesuré)",
    )

    # Speedup théorique (Amdahl avec f_serial moyen)
    f_mean = sub["f_serial"].mean()
    N_theo = np.linspace(sub["N"].min(), sub["N"].max(), 50)
    S_theo = amdahl_speedup(f_mean, P) * np.ones_like(N_theo)
    ax.plot(
        N_theo, S_theo,
        linestyle="--", color=colors.get(backend, "gray"),
        alpha=0.5, linewidth=1.5,
        label=f"{backend} (Amdahl f_s={f_mean:.2f})",
    )

ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.3)
ax.set_xlabel("N")
ax.set_ylabel("Speedup")
ax.set_title("Speedup mesuré vs Loi d'Amdahl")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_amdahl.png", dpi=150)
plt.show()
print("✅ plot_amdahl.png")


# ============================================================
# FIGURE 5 : Récapitulatif (4 sous-graphes)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- (1) Speedup ---
ax = axes[0, 0]
for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(sub["N"], sub["speedup"],
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "gray"),
            linewidth=2, markersize=6, label=backend)
ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
ax.axhline(y=P, color="gray", linestyle=":", alpha=0.3)
ax.set_title("Speedup S = T1 / T8")
ax.set_xlabel("N")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- (2) Efficiency ---
ax = axes[0, 1]
for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(sub["N"], sub["efficiency"],
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "gray"),
            linewidth=2, markersize=6, label=backend)
ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
ax.set_title(f"Efficacité E = S / {P}")
ax.set_xlabel("N")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- (3) Fraction parallèle ---
ax = axes[1, 0]
for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(sub["N"], sub["parallel_fraction"],
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "gray"),
            linewidth=2, markersize=6, label=backend)
ax.set_ylim(-0.05, 1.05)
ax.set_title("Fraction parallèle (Amdahl)")
ax.set_xlabel("N")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- (4) Fraction séquentielle ---
ax = axes[1, 1]
for backend in backends:
    sub = df[df["backend"] == backend]
    ax.plot(sub["N"], sub["f_serial"],
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "gray"),
            linewidth=2, markersize=6, label=backend)
ax.set_title("Fraction séquentielle (Amdahl)")
ax.set_xlabel("N")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("Analyse complète : Speedup, Efficacité, Loi d'Amdahl", fontsize=14)
plt.tight_layout()
plt.savefig("plot_summary.png", dpi=150)
plt.show()
print("✅ plot_summary.png")