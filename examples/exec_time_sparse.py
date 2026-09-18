"""Benchmark séquentiel : version Sparse avec 4 essais par N."""

from time import perf_counter
import csv
import numpy as np
import scipy.sparse as sp

from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


N_MEASURES = 4


def build_problem(n_nodes, avg_degree=10):
    rng = np.random.default_rng(42)
    n_edges = n_nodes * avg_degree

    edge_rows = rng.integers(0, n_nodes, size=n_edges).astype(np.int32)
    edge_cols = rng.integers(0, n_nodes, size=n_edges).astype(np.int32)
    edge_values = rng.random(n_edges).astype(np.float64)

    omega = LorentzianFrequencyGenerator(0.0, 1.0, False, 123).sample(
        n_nodes, truncated=True, cutoff=5.0
    )
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    model = SparseKuramotoModel(
        n_nodes=n_nodes, n_edges=n_edges,
        edge_values=edge_values, edge_rows=edge_rows, edge_cols=edge_cols,
        omega=omega, epsilon=3.8,
    )
    return model, theta0

def execution_time(model, theta0, T, dt, backend):
    start = perf_counter()
    time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    return perf_counter() - start


def save_csv(results, filename="execution_times_sparse_seq.csv"):
    header = ["N", "backend", "mean_s", "std_s", "min_s", "max_s", "n_measures"]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Sauvegardé : {filename}")


def main():
    dt = 0.01
    steps = 50
    T = steps * dt
    N_values = [50, 100, 200, 500, 800, 1000, 2000]
    backends = ["python", "cython", "c", "cpp"]


    print(f"BENCHMARK SPARSE ({N_MEASURES} essais par N)")


    results = []
    for N in N_values:
        print(f"\n N = {N}")
        model, theta0 = build_problem(N)

        for backend in backends:
            times = []
            for m in range(N_MEASURES):
                try:
                    elapsed = execution_time(model, theta0, T, dt, backend)
                    times.append(elapsed)
                except Exception as e:
                    print(f"  {backend} essai {m+1} : ERREUR - {e}")

            if times:
                mean_t = np.mean(times)
                std_t = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                results.append([N, backend, mean_t, std_t, min_t, max_t, len(times)])
                print(f"  {backend:<10} : {mean_t:.6f}s ± {std_t:.6f} "
                      f"(min={min_t:.6f}, max={max_t:.6f})")

    save_csv(results)


if __name__ == "__main__":
    main()