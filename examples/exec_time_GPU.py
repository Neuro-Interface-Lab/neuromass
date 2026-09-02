"""Test de performance : backend C avec/sans GPU - Cas Naive (dense)."""

import csv
import os
import time
import numpy as np
from neuromass.models.kuramoto import NaiveKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator

results_csv = "execution_times_GPU_dense_stats.csv"


def build_problem(n_nodes, density=0.1):
    """Construire un problème Kuramoto dense."""
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
    header = ["N", "mean_time", "std_time", "min_time", "max_time", "n_measures"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)
    print(f"\n Résultats sauvegardés dans : {filename}")


def main():
    # Paramètres
    T = 0.5
    dt = 0.01
    n_steps = int(T / dt)
    n_measures = 4

    # Différentes tailles à tester 
    N_values = [100, 200, 250, 255, 260,  500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]

   
    print("Test de performance : Cac naive (DENSE)")
   
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}")
    print(f"Nombre de mesures par N = {n_measures}\n")

    results = []

    for N in N_values:
        print(f"\n N = {N}")
        print("-" * 40)

        model, theta0 = build_problem(N)
        
        times = []

        for m in range(n_measures):
            try:
                elapsed = execution_time(model, theta0, T, dt, backend="c")
                times.append(elapsed)
                print(f"  Mesure {m+1}/{n_measures} : {elapsed:.6f}s")
            except Exception as e:
                print(f"  Mesure {m+1}/{n_measures} : ERREUR - {e}")
                times.append(None)

        valid_times = [t for t in times if t is not None]

        if len(valid_times) > 0:
            mean_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            
            print(f"\n Statistiques :")
            print(f"   Moyenne  : {mean_time:.6f}s")
            print(f"   Écart-type : {std_time:.6f}s")
            print(f"   Min      : {min_time:.6f}s")
            print(f"   Max      : {max_time:.6f}s")
            
            results.append([N, mean_time, std_time, min_time, max_time, len(valid_times)])
        else:
            print(f"  Aucune mesure valide pour N={N}")
            results.append([N, None, None, None, None, 0])

    # Sauvegarder les résultats
    save_results_to_csv(results)

 
  
    print("Rsumé de performance : Cas naive)")

    print(f"{'N':<12} {'Moyenne (s)':<16} {'Écart-type':<16} {'Min (s)':<14} {'Max (s)':<14} {'Mesures':<8}")
    print("-" * 85)
    for N, mean_t, std_t, min_t, max_t, n_meas in results:
        if mean_t is not None:
            print(f"{N:<12} {mean_t:<16.6f} ±{std_t:<15.6f} {min_t:<14.6f} {max_t:<14.6f} {n_meas:<8}")
        else:
            print(f"{N:<12} {'---':<16} {'---':<16} {'---':<14} {'---':<14} {n_meas:<8}")

    print("Test terminé")
    print("   - Modèle : Naive (dense)")
    print("   - Complexité théorique : O(N²)")

if __name__ == "__main__":
    main()