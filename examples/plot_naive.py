"""Trace les courbes du benchmark Naive Dense — SÉQUENTIEL."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_FILE = "execution_times_naive_seq.csv"

df = pd.read_csv(CSV_FILE)

# Styles par backend
styles = {
    "python": ("tab:blue",   "v", "--"),
    "cython": ("tab:orange", "s", "-"),
    "c":      ("tab:red",    "o", "-"),
    "cpp":    ("tab:green",  "D", "-"),
}

plt.figure(figsize=(6, 5))

for backend, (color, marker, ls) in styles.items():
    sub = df[df["backend"] == backend]
    if sub.empty:
        continue
    plt.loglog(
        sub["N"], sub["mean_s"],
        marker=marker, color=color, linestyle=ls,
        linewidth=2, markersize=8,
        label=backend,
    )

# Références asymptotiques
N_ref = np.array([50, 1000])
plt.loglog(N_ref, 1e-6 * N_ref,     "g--", alpha=0.4, label=r"$O(N)$")
plt.loglog(N_ref, 1e-8 * N_ref ** 2, "m:",  alpha=0.4, label=r"$O(N^2)$")

plt.xlabel("Nombre d'oscillateurs (N)")
plt.ylabel("Temps d'exécution (s)")
plt.title("Naive Dense : Séquentiel")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("benchmark_naive_seq.png", dpi=150)
plt.show()
print(" Sauvegardé : benchmark_naive_seq.png")