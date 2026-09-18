"""Benchmark and consistency check for the naive Kuramoto implementations."""

from time import perf_counter
import csv
import os
import numpy as np

from neuromass.models.kuramoto import NaiveKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator

TIMES_CSV = "execution_times_results.csv"


def build_problem(n_nodes, density=0.1):
    """Build a Kuramoto problem with n_nodes nodes and given density."""
    rng = np.random.default_rng(42)

    adjacency = rng.random((n_nodes, n_nodes))
    adjacency *= rng.random((n_nodes, n_nodes)) < density
    np.fill_diagonal(adjacency, 0.0)

    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0,
        gamma=1.0,
        symmetric=False,
        seed=123,
    )
    omega = frequency_generator.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    model = NaiveKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=3.8,
        adjacency=adjacency,
    )
    return model, theta0


def execution_time(model, theta0, T, dt, backend, omp_threads):
    """Measure the execution time of the Kuramoto model simulation."""
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    start = perf_counter()
    time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    elapsed = perf_counter() - start
    return elapsed


def save_results_to_csv(results, filename=TIMES_CSV):
    """Save the execution times to a CSV file."""
    file_exists = os.path.isfile(filename)
    header = ["N", "threads", "backend", "mean_time"]
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        for row in results:
            writer.writerow(row)



def main():
    # Paramètres
    dt = 0.01
    steps = 50
    T = steps * dt          # T = 0.5 s
    backends = ["c", "cpp"]  # Les deux backends à tester
    density = 1.0            # Matrice dense

    N_values = [50, 100, 200, 500, 800, 1000]
    threads_config = [1, 4, 8]

    print("=" * 60)
    print("BENCHMARK MAC : CAS DENSE")
    print(f"  - T = {T} s ({steps} pas de temps)")
    print(f"  - Backends : {backends}")
    print(f"  - Densité : {density}")
    print("=" * 60)

    results = []   # chaque élément : [N, threads, backend, temps_total, temps_par_pas, gain]

    for N in N_values:
        print(f"\n🔧 N = {N}")
        print("-" * 40)

        model, theta0 = build_problem(N, density=density)

        for backend in backends:
            print(f"\n  Backend : {backend}")
            times = {}   # threads -> temps

            for threads in threads_config:
                try:
                    elapsed = execution_time(model, theta0, T, dt, backend, threads)
                    times[threads] = elapsed
                    print(f"    {threads} thread(s) : {elapsed:.6f}s")
                except Exception as e:
                    times[threads] = None
                    print(f"    {threads} thread(s) : ERREUR - {e}")

            t1 = times.get(1)
            t4 = times.get(4)
            t8 = times.get(8)

            if t1 is not None and t8 is not None and t8 > 0:
                gain = t1 / t8
                temps_par_pas_1 = t1 / steps
                temps_par_pas_4 = t4 / steps if t4 is not None else 0
                temps_par_pas_8 = t8 / steps

                print(f"\n    Gain (1→8 threads) : ×{gain:.2f}")
                print(f"    Temps par pas (1 thread)  : {temps_par_pas_1:.6f}s")
                if t4 is not None:
                    print(f"    Temps par pas (4 threads) : {temps_par_pas_4:.6f}s")
                print(f"    Temps par pas (8 threads) : {temps_par_pas_8:.6f}s")

                # Sauvegarde
                results.append([N, 1, backend, t1, temps_par_pas_1, ""])
                if t4 is not None:
                    results.append([N, 4, backend, t4, temps_par_pas_4, ""])
                results.append([N, 8, backend, t8, temps_par_pas_8, f"{gain:.2f}"])

    # Sauvegarder
    save_results_to_csv(results)
    print(f"\n Résultats sauvegardés dans : {TIMES_CSV}")

    # Affichage résumé
    
    print("RÉSUMÉ DES PERFORMANCES (CAS DENSE)")
    print(f"{'N':<8} {'Threads':<8} {'Backend':<10} {'Temps total (s)':<16} {'Temps/pas (s)':<14} {'Gain':<8}")
    print("-" * 75)
    for row in results:
        N, threads, backend, t_total, t_pas, gain = row
        print(f"{N:<8} {threads:<8} {backend:<10} {t_total:<16.6f} {t_pas:<14.6f} {gain:<8}")

if __name__ == "__main__":
    main()