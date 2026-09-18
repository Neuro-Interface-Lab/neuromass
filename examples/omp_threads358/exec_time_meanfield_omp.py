"""Benchmark OpenMP Mean-Field : 3, 5, 8 threads dans un seul CSV."""

from time import perf_counter
import csv
import numpy as np

from neuromass.models.kuramoto import MeanFieldKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


N_MEASURES = 4
THREAD_COUNTS = [3, 5, 8]
N_values = [50, 100, 200, 500, 800, 1000, 2000]
BACKENDS = ["python", "cython", "c", "cpp"]


def build_problem(n_nodes):
    """Construit un problème Mean-Field Kuramoto."""
    rng = np.random.default_rng(42)

    freq_gen = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123
    )
    omega = freq_gen.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    model = MeanFieldKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=3.8,
    )
    return model, theta0


def execution_time(model, theta0, T, dt, backend):
    start = perf_counter()
    time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    return perf_counter() - start


def save_csv(results, filename="execution_times_meanfield_omp_threads.csv"):
    """Sauvegarde tous les résultats dans UN SEUL CSV."""
    header = ["N", "backend", "n_threads", "mean_s", "std_s",
              "min_s", "max_s", "n_measures"]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nSauvegardé : {filename}")


def main():
    import os

    dt = 0.01
    steps = 50
    T = steps * dt

    print("=" * 70)
    print(f"  BENCHMARK MEAN-FIELD MULTI-THREADS")
    print(f"  Threads testés : {THREAD_COUNTS}")
    print("=" * 70)

    results = []

    for n_threads in THREAD_COUNTS:
        # Doit être défini AVANT le premier import natif pour être pris
        # en compte. Ici on le force dans l'environnement courant, mais
        # OpenMP l'a peut-être déjà lu au premier import.
        os.environ["OMP_NUM_THREADS"] = str(n_threads)

        print(f"\n{'=' * 70}")
        print(f"  OMP_NUM_THREADS = {n_threads}")
        print(f"{'=' * 70}")

        for N in N_values:
            print(f"\n N = {N}")
            model, theta0 = build_problem(N)

            for backend in BACKENDS:
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
                    results.append([
                        N, backend, n_threads,
                        mean_t, std_t, min_t, max_t, len(times)
                    ])
                    print(f"  {backend:<12} : {mean_t:.6f}s ± {std_t:.6f} "
                          f"(min={min_t:.6f}, max={max_t:.6f})")

    save_csv(results)


if __name__ == "__main__":
    main()