"""Benchmark GPU Sparse CSR avec d proportionnel à N².

Loi de scaling : d(N) = K₂ * N²
  → M = N * d(N) = K₂ * N³  (croissance très agressive)
  → Complexité attendue : O(N³)
"""

import csv
import time
import numpy as np

from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


# PARAMÈTRES GLOBAUX


RESULTS_CSV = "execution_times_GPU_sparse_quadratic.csv"

DEGREE_REF = 10          # degré souhaité à N = N_REF
N_REF = 1000             # taille de référence
K2 = DEGREE_REF / (N_REF ** 2)   # K2 = 10 / 1e6 = 1e-5

T = 0.5
DT = 0.01
N_MEASURES = 3
BACKEND = "c"            # GPU

# Tailles adaptées au cas O(N³)
# M = K2 * N³ explose TRÈS vite (voir table ci-dessous)
N_VALUES =[500, 1000, 5000, 10000, 20000, 30000, 40000 ]


# LOI DE SCALING DU DEGRÉ


def degree_for(n_nodes):
    """d(N) = K2 * N², arrondi, minimum 1."""
    return max(1, round(K2 * n_nodes ** 2))



# CONSTRUCTION DU PROBLÈME


def build_sparse_problem(n_nodes, degree):
    """Construit un problème Kuramoto sparse (CSR)."""
    rng = np.random.default_rng(42)
    n_edges = n_nodes * degree

    edge_rows = np.repeat(np.arange(n_nodes, dtype=np.int32), degree)

    edge_cols = rng.integers(0, n_nodes, size=n_edges, dtype=np.int32)
    same = edge_cols == edge_rows
    while np.any(same):
        edge_cols[same] = rng.integers(0, n_nodes, size=same.sum(), dtype=np.int32)
        same = edge_cols == edge_rows

    edge_values = np.ones(n_edges, dtype=np.float32)

    freq_gen = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123,
    )
    omega = freq_gen.sample(n_nodes, truncated=True, cutoff=5.0).astype(np.float32)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes).astype(np.float32)

    model = SparseKuramotoModel(
        n_nodes=n_nodes, n_edges=n_edges,
        edge_values=edge_values, edge_rows=edge_rows, edge_cols=edge_cols,
        omega=omega, epsilon=3.8,
    )
    return model, theta0


def execution_time(model, theta0, T, dt, backend):
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
    print(f"\nRésultats sauvegardés dans : {filename}")



# MAIN


def main():

    print("  BENCHMARK GPU SPARSE CSR — d ∝ N²")

    print(f"  dt = {DT}, T = {T}, mesures = {N_MEASURES}")
    print(f"  Backend = {BACKEND} (GPU)")
    print(f"  d(N) = K2 * N²  avec K2 = {K2}")
    print(f"  M = K2 * N³")
   

    results = []

    for N in N_VALUES:
        degree = degree_for(N)
        n_edges = N * degree
        density = n_edges / (N ** 2)

        print(f"\n N = {N}  |  d = {degree}  |  M = {n_edges}  |  densité = {density:.4f}")

        try:
            model, theta0 = build_sparse_problem(N, degree=degree)
        except Exception as e:
            print(f"  Erreur : {e}")
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
            print(f"  → moyenne = {mean_t:.6f}s ± {std_t:.6f}")
            results.append([N, degree, n_edges, density,
                            mean_t, std_t, np.min(times), np.max(times), len(times)])
        else:
            results.append([N, degree, n_edges, density, None, None, None, None, 0])

    save_results_to_csv(results)

    # Résumé
  
    print("  RÉSUMÉ")
  
    print(f"{'N':<10}{'d(N)':<10}{'M':<16}{'densité':<12}{'temps (s)':<14}")

    for row in results:
        N, degree, n_edges, density, mean_t = row[:5]
        if mean_t is not None:
            print(f"{N:<10}{degree:<10}{n_edges:<16}{density:<12.4f}{mean_t:<14.6f}")
        else:
            print(f"{N:<10}{degree:<10}{n_edges:<16}{density:<12.4f}{'---':<14}")


if __name__ == "__main__":
    main()