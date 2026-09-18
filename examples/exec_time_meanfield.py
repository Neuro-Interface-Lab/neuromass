"""Benchmark séquentiel : version Mean-Field avec 4 essais par N."""

from time import perf_counter
import csv
import numpy as np

from neuromass.models.kuramoto import MeanFieldKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


N_MEASURES = 4  # Nombre d'essais par taille


def build_problem(n_nodes):
    """Construit un problème Mean-Field Kuramoto."""
    rng = np.random.default_rng(42)

    # Génération des fréquences propres (Lorentzienne)
    freq_gen = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123
    )
    omega = freq_gen.sample(n_nodes, truncated=True, cutoff=5.0)

    # Phases initiales
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    # Modèle Mean-Field
    model = MeanFieldKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=3.8,
    )

    return model, theta0


def execution_time(model, theta0, T, dt, backend):
    """Mesure le temps d'exécution pour un backend donné."""
    start = perf_counter()
    time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    return perf_counter() - start


def save_csv(results, filename="execution_times_meanfield_seq.csv"):
    """Sauvegarde les résultats dans un fichier CSV."""
    header = ["N", "backend", "mean_s", "std_s", "min_s", "max_s", "n_measures"]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nSauvegardé : {filename}")


def main():
    # Paramètres temporels
    dt = 0.01
    steps = 50
    T = steps * dt

    # Tailles testées
    N_values = [50, 100, 200, 500, 800, 1000, 2000]

    # Backends à comparer
    backends = ["python", "cython", "c", "cpp"]

    print("=" * 70)
    print(f"  BENCHMARK MEAN-FIELD ({N_MEASURES} essais par N)")
    print("=" * 70)

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