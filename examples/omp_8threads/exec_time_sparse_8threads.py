"""Benchmark Sparse CSR : 8 threads."""

import os
os.environ["OMP_NUM_THREADS"] = "8"

from time import perf_counter
import csv
import numpy as np

from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


N_MEASURES = 4
N_THREADS = 8
N_values = [50, 100, 200, 500, 800, 1000, 2000]
BACKENDS = ["cython", "c", "cpp"]


def build_problem(n_nodes, avg_degree=10):
    rng = np.random.default_rng(42)
    n_edges = n_nodes * avg_degree
    edge_rows = rng.integers(0, n_nodes, size=n_edges).astype(np.int32)
    edge_cols = rng.integers(0, n_nodes, size=n_edges).astype(np.int32)
    edge_values = rng.random(n_edges).astype(np.float64)

    freq_gen = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123
    )
    omega = freq_gen.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    model = SparseKuramotoModel(
        n_nodes=n_nodes, n_edges=n_edges,
        edge_values=edge_values, edge_rows=edge_rows, edge_cols=edge_cols,
        omega=omega, epsilon=3.8,
    )
    return model, theta0


def execution_time(model, theta0, T, dt, backend):
    start = perf_counter()
    model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    return perf_counter() - start


def save_csv(results, filename="execution_times_sparse_8threads.csv"):
    header = ["N", "backend", "n_threads", "mean_s", "std_s",
              "min_s", "max_s", "n_measures"]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\n💾 Sauvegardé : {filename}")


def main():
    dt = 0.01
    steps = 50
    T = steps * dt

    print("=" * 70)
    print(f"  BENCHMARK SPARSE CSR — {N_THREADS} THREADS")
    print("=" * 70)

    results = []
    for N in N_values:
        print(f"\n N = {N}")
        model, theta0 = build_problem(N)
        for backend in BACKENDS:
            times = []
            for m in range(N_MEASURES):
                try:
                    times.append(execution_time(model, theta0, T, dt, backend))
                except Exception as e:
                    print(f"  {backend} essai {m+1} : ERREUR - {e}")
            if times:
                results.append([
                    N, backend, N_THREADS,
                    np.mean(times), np.std(times),
                    np.min(times), np.max(times), len(times)
                ])
                print(f"  {backend:<12} : {np.mean(times):.6f}s ± {np.std(times):.6f} "
                      f"(min={np.min(times):.6f}, max={np.max(times):.6f})")

    save_csv(results)


if __name__ == "__main__":
    main()