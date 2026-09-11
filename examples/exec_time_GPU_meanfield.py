import csv
import os
import time
import numpy as np
from neuromass.models.kuramoto import MeanFieldKuramotoModel  
from neuromass.utils import LorentzianFrequencyGenerator

results_csv = "execution_times_meanfield_stats.csv"  


def build_meanfield_problem(n_nodes):
    """
    Construire un problème Kuramoto avec couplage mean-field (paramètre complexe).
    Pas de matrice d'adjacence, complexité O(N).
    """
    rng = np.random.default_rng(42)
    
    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0,
        gamma=1.0,
        symmetric=False,
        seed=123,
    )
    omega = frequency_generator.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)
    

    model = MeanFieldKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=3.8,
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
    print(f"\nRésultats sauvegardés dans : {filename}")


def main():
    # Paramètres
    T = 0.5
    dt = 0.01
    n_steps = int(T / dt)
    n_measures = 4


    N_values = [1000, 2000, 5000, 10000, 20000, 50000, 
                100000, 200000, 500000, 1000000, 2000000, 5000000]  # Valeurs de N à tester

  
    print("TEST DE PERFORMANCE : MEAN-FIELD (PARAMÈTRE COMPLEXE)")
   
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}")
    print(f"Nombre de mesures par N = {n_measures}")
    print(f"Complexité théorique : O(N)\n")

    results = []

    for N in N_values:
        print(f"\nN = {N}")
        print("-" * 40)

        # Construire le modèle mean-field
        model, theta0 = build_meanfield_problem(N)
        
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
            
            print(f"\n  Statistiques :")
            print(f"    Moyenne  : {mean_time:.6f}s")
            print(f"    Écart-type : {std_time:.6f}s")
            print(f"    Min      : {min_time:.6f}s")
            print(f"    Max      : {max_time:.6f}s")
            
            results.append([N, mean_time, std_time, min_time, max_time, len(valid_times)])
        else:
            print(f"  Aucune mesure valide pour N={N}")
            results.append([N, None, None, None, None, 0])

    # Sauvegarder les résultats
    save_results_to_csv(results)

    # Résumé
 
    print("RÉSUMÉ DES PERFORMANCES (MEAN-FIELD)")
 
    print(f"{'N':<12} {'Moyenne (s)':<16} {'Écart-type':<16} {'Min (s)':<14} {'Max (s)':<14} {'Mesures':<8}")
    print("-" * 85)
    for N, mean_t, std_t, min_t, max_t, n_meas in results:
        if mean_t is not None:
            print(f"{N:<12} {mean_t:<16.6f} ±{std_t:<15.6f} {min_t:<14.6f} {max_t:<14.6f} {n_meas:<8}")
        else:
            print(f"{N:<12} {'---':<16} {'---':<16} {'---':<14} {'---':<14} {n_meas:<8}")


    print("   - Modèle : Mean-Field (paramètre d'ordre global)")
    print("   - Complexité théorique : O(N)")
    
  


if __name__ == "__main__":
    main()