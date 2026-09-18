"""Trace les 4 backends de la version Sparse avec références O(N) et O(N²)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sparse_all_backends(
    csv_file="execution_times_sparse_seq.csv",
    output_file="courbe_sparse_all_backends.png"
):
    """
    Trace les 4 backends (python, cython, c, cpp) de la version Sparse
    avec les droites de référence O(N) et O(N²).
    """
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(6, 5))

    # Couleurs pour chaque backend
    colors = {
        "python": "tab:blue",
        "cython": "tab:orange",
        "c": "tab:red",
        "cpp": "tab:green",
    }

    # Marqueurs
    markers = {
        "python": "o",
        "cython": "s",
        "c": "^",
        "cpp": "D",
    }

    # =========================================
    # Tracer chaque backend
    # =========================================
    for backend in ["python", "cython", "c", "cpp"]:
        subset = df[df["backend"] == backend].sort_values("N")
        if subset.empty:
            continue

        plt.errorbar(
            subset["N"],
            subset["mean_s"],
            yerr=subset["std_s"],
            fmt=f"{markers[backend]}-",
            label=backend.upper(),
            linewidth=2,
            markersize=8,
            capsize=4,
            color=colors[backend],
        )

    # =========================================
    # Droites de référence (basées sur le backend C)
    # =========================================
    ref_subset = df[df["backend"] == "c"].sort_values("N")
    if not ref_subset.empty:
        N = ref_subset["N"].values
        mean_t = ref_subset["mean_s"].values

        N_ref = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)

        # O(N)
        a_N = mean_t[0] / N[0]
        plt.plot(N_ref, a_N * N_ref, "--", label=r"$O(N)$", color="green", alpha=0.7, linewidth=2)

        # O(N²)
        b_N2 = mean_t[0] / (N[0] ** 2)
        plt.plot(N_ref, b_N2 * N_ref**2, ":", label=r"$O(N^2)$", color="purple", alpha=0.7, linewidth=2)

    # =========================================
    # Mise en forme
    # =========================================
    plt.xlabel("Nombre d'oscillateurs (N)", fontsize=13)
    plt.ylabel("Temps d'exécution (s)", fontsize=13)
    plt.title("Sparse : Comparaison des 4 backends avec références O(N) et O(N²)", fontsize=14)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3, which="both")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f" Courbe sauvegardée : {output_file}")


def main():
    plot_sparse_all_backends(
        csv_file="execution_times_sparse_seq.csv",
        output_file="courbe_sparse_all_backends.png",
    )


if __name__ == "__main__":
    main()