"""Benchmark GPU Sparse CSR avec d proportionnel à N.

Loi de scaling : d(N) = K * N
  → M = N * d(N) = K * N² 
  → Complexité attendue : O(N²)
"""

import csv
import time
import numpy as np

from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


# PARAMÈTRES GLOBAUX


RESULTS_CSV = "execution_times_GPU_sparse_linear.csv"

DEGREE_REF = 10        # degré souhaité à N = N_REF
N_REF = 1000           # taille de référence
K = DEGREE_REF / N_REF # coefficient de proportionnalité (K = 0.01)

T = 0.5
DT = 0.01
N_MEASURES = 3
BACKEND = "c"          # backend GPU 

# Tailles à tester
# ATTENTION : M = K*N² explose rapidement. On limite donc les N.
N_VALUES = [500, 1000, 5000, 10000, 20000, 30000, 40000, 100000,300000,500000 ]


# LOI DE SCALING DU DEGRÉ


def degree_for(n_nodes):
    """d(N) = K * N, arrondi, minimum 1."""
    return max(1, round(K * n_nodes))



# CONSTRUCTION DU PROBLEME


def build_sparse_problem(n_nodes, degree):
    """Construit un problème Kuramoto sparse (CSR) avec un degré donné."""
    rng = np.random.default_rng(42)
    n_edges = n_nodes * degree

    # Lignes : chaque nœud apparaît `degree` fois
    edge_rows = np.repeat(np.arange(n_nodes, dtype=np.int32), degree)

    # Colonnes : voisins aléatoires différents du nœud lui-même
    edge_cols = rng.integers(0, n_nodes, size=n_edges, dtype=np.int32)
    same = edge_cols == edge_rows
    while np.any(same):
        edge_cols[same] = rng.integers(0, n_nodes, size=same.sum(), dtype=np.int32)
        same = edge_cols == edge_rows

    edge_values = np.ones(n_edges, dtype=np.float32)

    # Fréquences
    freq_gen = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123,
    )
    omega = freq_gen.sample(n_nodes, truncated=True, cutoff=5.0).astype(np.float32)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes).astype(np.float32)

    model = SparseKuramotoModel(
        n_nodes=n_nodes,
        n_edges=n_edges,
        edge_values=edge_values,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        omega=omega,
        epsilon=3.8,
    )
    return model, theta0


def execution_time(model, theta0, T, dt, backend):
    """Mesure le temps d'exécution."""
    start = time.perf_counter()
    model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    return time.perf_counter() - start


def save_results_to_csv(results, filename=RESULTS_CSV):
    header = ["N", "degree", "n_edges", "density",
              "mean_time", "std_time", "min_time", "max_time", "n_measures"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\n Résultats sauvegardés dans : {filename}")



# MAIN


def main():
   
    print("  BENCHMARK GPU SPARSE CSR — d ∝ N")
    print(f"  dt = {DT}, T = {T}, mesures par point = {N_MEASURES}")
    print(f"  Backend = {BACKEND} (GPU)")
    print(f"  d(N) = K * N  avec K = {K}")
    print(f"  M = K * N²  (densité constante = {K})")


    results = []

    for N in N_VALUES:
        degree = degree_for(N)
        n_edges = N * degree
        density = n_edges / (N ** 2)

        print(f"\n N = {N}  |  d = {degree}  |  M = {n_edges}  |  densité = {density:.4f}")

        try:
            model, theta0 = build_sparse_problem(N, degree=degree)
        except Exception as e:
            print(f"  Erreur de construction : {e}")
            results.append([N, degree, n_edges, density, None, None, None, None, 0])
            continue

        times = []
        for m in range(N_MEASURES):
            try:
                elapsed = execution_time(model, theta0, T, DT, backend=BACKEND)
                times.append(elapsed)
                print(f"  Mesure {m+1}/{N_MEASURES} : {elapsed:.6f}s")
            except Exception as e:
                print(f"  Mesure {m+1}/{N_MEASURES} : ERREUR - {e}")

        if times:
            mean_t = np.mean(times)
            std_t = np.std(times)
            min_t = np.min(times)
            max_t = np.max(times)
            print(f"  → moyenne = {mean_t:.6f}s ± {std_t:.6f}")
            results.append([N, degree, n_edges, density,
                            mean_t, std_t, min_t, max_t, len(times)])
        else:
            results.append([N, degree, n_edges, density, None, None, None, None, 0])

    save_results_to_csv(results)

    # Résumé

    print("  RÉSUMÉ")
    print(f"{'N':<10}{'d(N)':<8}{'M':<14}{'densité':<10}{'temps moyen (s)':<16}")
  
    for row in results:
        N, degree, n_edges, density, mean_t = row[:5]
        if mean_t is not None:
            print(f"{N:<10}{degree:<8}{n_edges:<14}{density:<10.4f}{mean_t:<16.6f}")
        else:
            print(f"{N:<10}{degree:<8}{n_edges:<14}{density:<10.4f}{'---':<16}")

    print("\nTest terminé.")


if __name__ == "__main__":
    main()