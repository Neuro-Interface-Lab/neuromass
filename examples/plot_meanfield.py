"""Trace les courbes du benchmark Mean-Field."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Charger le CSV
df = pd.read_csv("execution_times_meanfield_seq.csv")

# Couleurs par backend
colors = {
    "python": "tab:blue",
    "cython": "tab:orange",
    "c": "tab:red",
    "cpp": "tab:green",
}
markers = {
    "python": "o",
    "cython": "s",
    "c": "^",
    "cpp": "D",
}


plt.figure(figsize=(6, 5))

# Courbes des backends
for backend in ["python", "cython", "c", "cpp"]:
    sub = df[df["backend"] == backend]
    plt.loglog(
        sub["N"], sub["mean_s"],
        marker=markers[backend],
        color=colors[backend],
        label=backend.upper(),
        linewidth=2,
        markersize=8,
    )

# Références O(N) et O(N²)
N_ref = np.array([50, 2000])
ref_O_N = 1e-5 * N_ref
ref_O_N2 = 1e-8 * N_ref ** 2

plt.loglog(N_ref, ref_O_N, "g--", label=r"$O(N)$", alpha=0.5)
plt.loglog(N_ref, ref_O_N2, "m:", label=r"$O(N^2)$", alpha=0.5)

plt.xlabel("Nombre d'oscillateurs (N)")
plt.ylabel("Temps d'exécution (s)")
plt.title("Mean-Field : Comparaison des 4 backends")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("benchmark_meanfield_seq.png", dpi=150)
plt.show()