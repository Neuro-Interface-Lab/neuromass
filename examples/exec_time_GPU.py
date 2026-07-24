"""Test de performance : backend C avec/sans GPU."""
import csv
import os
import time
import numpy as np
from neuromass.models.kuramoto import NaiveKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator

results_csv = "execution_times_GPU.csv"

def build_problem(n_nodes, density=0.1):
    """Construire un problème Kuramoto."""
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

def execution_time(model, theta0, T, dt, backend):
    """Mesurer le temps d'exécution."""
    start = time.perf_counter()
    time_arr, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    elapsed = time.perf_counter() - start
    return elapsed

def save_results_to_csv(results, filename=results_csv):
    """Sauvegarder les résultats dans un fichier CSV."""
    file_exists = os.path.isfile(filename)
    header = ["N", "time", "note"]
    
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)
    
    print(f"\nRésultats sauvegardés dans : {filename}")


def main():
    # Paramètres
    T = 0.5
    dt = 0.01
    n_steps = int(T / dt)

    # Différentes tailles à tester
    N_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

   
    print("TEST DE PERFORMANCE : BACKEND C (CPU/GPU)")
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}\n")

    results = []  # chaque élément : (N, elapsed, note)

    for N in N_values:
        print(f"\n N = {N}")
        

        model, theta0 = build_problem(N)

        try:
            elapsed = execution_time(model, theta0, T, dt, backend="c")
            print(f"  backend='c' : {elapsed:.6f}s")



            results.append([N, elapsed])

        except Exception as e:
            print(f"  backend='c' : ERREUR - {e}")
            results.append([N, None, f"ERREUR: {e}"])

    # Sauvegarder les résultats
    save_results_to_csv(results)

    # Résumé
   
    print("RÉSUMÉ DES PERFORMANCES")

    print(f"{'N':<10} {'Temps (s)':<15} {'Notes':<20}")


    for N, t in results:
        if t is not None:
            print(f"{N:<10} {t:<15.6f} ")
        else:
            print(f"{N:<10} {'---':<15} ")

   
    print("Test terminé")
    print("   - Pour N < 1000 : OpenMP (CPU)")
    print("   - Pour N > 10000 : CUDA (GPU)")
 
  


if __name__ == "__main__":
    main()

