"""Test de performance : backend C avec/sans GPU - Version CSR (matrice creuse)."""

import csv
import os
import time
import numpy as np
from neuromass.models.kuramoto import SparseKuramotoModel  
from neuromass.utils import LorentzianFrequencyGenerator

results_csv = "execution_times_CSR_stats.csv"  


def build_sparse_problem(n_nodes, degree=50):
    """
    Construire un problème Kuramoto avec une matrice creuse (CSR).
    degree = nombre moyen de voisins par oscillateur.
    """
    rng = np.random.default_rng(42)
    
    n_edges = n_nodes * degree
    
    edge_rows = np.zeros(n_edges, dtype=np.int32)
    edge_cols = np.zeros(n_edges, dtype=np.int32)
    edge_values = np.ones(n_edges, dtype=np.float32)
    
    for i in range(n_nodes):
        for k in range(degree):
            idx = i * degree + k
            edge_rows[idx] = i
            j = rng.integers(0, n_nodes)
            while j == i:
                j = rng.integers(0, n_nodes)
            edge_cols[idx] = j
    
    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0,
        gamma=1.0,
        symmetric=False,
        seed=123,
    )
    omega = frequency_generator.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)
    
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
    """Mesurer le temps d'exécution."""
    start = time.perf_counter()
    time_arr, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    elapsed = time.perf_counter() - start
    return elapsed


def save_results_to_csv(results, filename=results_csv):
    """Sauvegarder les résultats dans un fichier CSV."""
    header = ["N", "degree", "mean_time", "std_time", "min_time", "max_time", "n_measures"]
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
    
    # Degré moyen pour la matrice creuse
    degree = 50  # Chaque nœud a 50 voisins en moyenne

    # Différentes tailles à tester 
    N_values = [500, 1000, 2000, 5000, 10000, 20000, 50000,80000, 100000, 200000,600000,800000, 1000000]

    print("=" * 60)
    print("TEST DE PERFORMANCE : BACKEND C (CSR)")
    print("=" * 60)
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}")
    print(f"Nombre de mesures par N = {n_measures}")
    print(f"Degré moyen = {degree} (M = N × {degree})\n")

    results = []

    for N in N_values:
        print(f"\nN = {N}")
        print("-" * 40)

        # Construire le modèle CSR
        model, theta0 = build_sparse_problem(N, degree=degree)
        
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
            
            mem = N * degree * 4 / 1e6  # Mémoire en Mo (float)
            print(f"\n  Statistiques :")
            print(f"    Moyenne  : {mean_time:.6f}s")
            print(f"    Écart-type : {std_time:.6f}s")
            print(f"    Min      : {min_time:.6f}s")
            print(f"    Max      : {max_time:.6f}s")
            print(f"    Mémoire CSR : {mem:.2f} Mo")
            
            results.append([N, degree, mean_time, std_time, min_time, max_time, len(valid_times)])
        else:
            print(f"   Aucune mesure valide pour N={N}")
            results.append([N, degree, None, None, None, None, 0])

    # Sauvegarder les résultats
    save_results_to_csv(results)

    
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES PERFORMANCES (CSR)")
    print("=" * 70)
    print(f"{'N':<10} {'Degré':<8} {'Moyenne (s)':<15} {'Écart-type':<15} {'Min (s)':<12} {'Max (s)':<12} {'Mesures':<8}")
    print("-" * 85)
    for N, deg, mean_t, std_t, min_t, max_t, n_meas in results:
        if mean_t is not None:
            print(f"{N:<10} {deg:<8} {mean_t:<15.6f} ±{std_t:<14.6f} {min_t:<12.6f} {max_t:<12.6f} {n_meas:<8}")
        else:
            print(f"{N:<10} {deg:<8} {'---':<15} {'---':<15} {'---':<12} {'---':<12} {n_meas:<8}")

    print("\n" + "=" * 60)
    print("Test terminé")
    print(f"   - Format CSR (matrice creuse)")
    print(f"   - Degré moyen = {degree}")
    print("=" * 60)


if __name__ == "__main__":
    main()