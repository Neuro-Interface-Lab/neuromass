import csv
import os
import numpy as np
 
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[info] matplotlib non installe -> le graphique sera saute (pip install matplotlib)")
 
 

# Lecture des CSV (remplace pandas.read_csv)

def read_csv_dict(path, n_col, t_col):
    """Lit un CSV et retourne un dict {N: temps} pour les colonnes demandees."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(float(row[n_col]))
            t = float(row[t_col])
            data[N] = t
    return data
 
 

# Calcul du speedup sur les N communs (remplace pandas.merge)

def compute_speedup(t_ref, t_new):
    """t_ref, t_new : dict {N: temps}. Retourne N_array, speedup_array (tries par N)."""
    common_N = sorted(set(t_ref.keys()) & set(t_new.keys()))
    N_arr = np.array(common_N, dtype=float)
    speedup = np.array([t_ref[n] / t_new[n] for n in common_N])
    return N_arr, speedup
 
 



if __name__ == "__main__":
 
    t_mf = read_csv_dict("execution_times_meanfield_stats.csv", "N", "mean_time")
    t_csr = read_csv_dict("execution_times_CSR_stats.csv", "N", "mean_time")
    t_gpu = read_csv_dict("execution_times_GPU.csv", "N", "time")
 
    #  Speedup Mean-Field vs CSR
    N_csr, speedup_csr = compute_speedup(t_csr, t_mf)
    print("=== Gain Mean-Field vs CSR (CPU) ===")
    print(f"{'N':>10} {'t_csr':>12} {'t_meanfield':>14} {'speedup':>10}")
    for n, s in zip(N_csr, speedup_csr):
        print(f"{int(n):>10} {t_csr[int(n)]:>12.6f} {t_mf[int(n)]:>14.6f} {s:>10.3f}")

    #  Speedup Mean-Field vs GPU naive 
    N_gpu, speedup_gpu = compute_speedup(t_gpu, t_mf)
    print("\n=== Gain Mean-Field vs GPU naive ===")
    print(f"{'N':>10} {'t_gpu':>12} {'t_meanfield':>14} {'speedup':>10}")
    for n, s in zip(N_gpu, speedup_gpu):
        print(f"{int(n):>10} {t_gpu[int(n)]:>12.6f} {t_mf[int(n)]:>14.6f} {s:>10.3f}")
 
    # Graphique 
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
        plots = [
            (axes[0], N_csr, speedup_csr, "Mean-Field vs CSR (CPU)"),
            (axes[1], N_gpu, speedup_gpu, "Mean-Field vs GPU naive"),
        ]
        for ax, N, speedup, title in plots:
            ax.plot(N, speedup, "o-", color="blue", label="Speedup mesurés")
            ax.set_xscale("log")
            ax.set_xlabel("N (nombre d'oscillateurs)")
            ax.set_ylabel("Speedup (x)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)
 
        plt.tight_layout()
        out_path = "speedup_gains.jpg"
        plt.savefig(out_path, dpi=150)
        print(f"\nGraphique enregistre : {out_path}")