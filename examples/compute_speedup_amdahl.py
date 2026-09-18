"""Calcule le speedup et estime la fraction parallèle (loi d'Amdahl)."""

import os
import pandas as pd
import numpy as np


CONFIGS = {
    "Naive Dense": {
        "seq": "execution_times_naive_1threads.csv",
        "omp": "execution_times_naive_8threads.csv",
        "out": "speedup_naive.csv",
    },
    "Sparse CSR": {
        "seq": "execution_times_sparse_1threads.csv",
        "omp": "execution_times_sparse_8threads.csv",
        "out": "speedup_sparse.csv",
    },
    "Mean-Field": {
        "seq": "execution_times_meanfield_1threads.csv",
        "omp": "execution_times_meanfield_8threads.csv",
        "out": "speedup_meanfield.csv",
    },
}

BACKENDS = ["cython", "c", "cpp"]
P = 8   # Nombre de threads


def amdahl_speedup(f, p):
    """Loi d'Amdahl : speedup théorique pour fraction parallèle f."""
    return 1.0 / ((1.0 - f) + f / p)


def estimate_f_from_speedup(S, p):
    """Estime la fraction parallèle f à partir du speedup mesuré S."""
    if S <= 1:
        return 0.0
    return (1.0 - 1.0 / S) / (1.0 - 1.0 / p)


def compute_speedup_amdahl(seq_file, omp_file, out_file):
    if not os.path.exists(seq_file) or not os.path.exists(omp_file):
        print(f"⚠️  fichiers manquants pour {out_file}")
        return None

    df_seq = pd.read_csv(seq_file)
    df_omp = pd.read_csv(omp_file)

    df_seq = df_seq[df_seq["backend"].isin(BACKENDS)]
    df_omp = df_omp[df_omp["backend"].isin(BACKENDS)]

    merged = pd.merge(
        df_seq[["N", "backend", "mean_s"]],
        df_omp[["N", "backend", "mean_s"]],
        on=["N", "backend"],
        suffixes=("_1T", "_8T"),
    )

    # Speedup mesuré
    merged["speedup"] = merged["mean_s_1T"] / merged["mean_s_8T"]

    # Estimation de f (fraction parallèle) via Amdahl
    merged["f_estimated"] = merged["speedup"].apply(
        lambda S: estimate_f_from_speedup(S, P)
    )

    # Speedup théorique max (p → ∞)
    merged["speedup_max"] = merged["f_estimated"].apply(
        lambda f: 1.0 / (1.0 - f) if f < 1 else np.inf
    )

    # Speedup théorique avec 8 threads (vérification)
    merged["speedup_theorique_8T"] = merged["f_estimated"].apply(
        lambda f: amdahl_speedup(f, P)
    )

    merged.to_csv(out_file, index=False)
    print(f"\n💾 {out_file}")
    print(merged.to_string(index=False))
    return merged


def main():
    all_results = {}
    for name, cfg in CONFIGS.items():
        print("=" * 70)
        print(f"  {name}")
        print("=" * 70)
        result = compute_speedup_amdahl(cfg["seq"], cfg["omp"], cfg["out"])
        if result is not None:
            all_results[name] = result

    # Résumé
    print("\n" + "=" * 70)
    print("  RÉSUMÉ — Loi d'Amdahl")
    print("=" * 70)
    for name, df in all_results.items():
        print(f"\n{name} :")
        for backend in BACKENDS:
            sub = df[df["backend"] == backend]
            if sub.empty:
                continue
            f_mean = sub["f_estimated"].mean()
            S_mean = sub["speedup"].mean()
            S_max = 1.0 / (1.0 - f_mean) if f_mean < 1 else np.inf
            print(f"  {backend:<8} : "
                  f"speedup moyen = {S_mean:.2f}×  |  "
                  f"f ≈ {f_mean:.3f}  |  "
                  f"speedup max théorique ≈ {S_max:.1f}×")


if __name__ == "__main__":
    main()