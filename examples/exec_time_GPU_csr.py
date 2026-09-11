import csv
import time
import numpy as np
from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator

results_csv = "execution_times_GPU_sparse_stats.csv"


def build_sparse_problem(n_nodes, degree=10):
    """
    Construire un problème Kuramoto sparse avec un degré fixe.
    Évite la construction de la matrice dense N×N.
    """
    rng = np.random.default_rng(42)
    
    # Nombre d'arêtes
    n_edges = n_nodes * degree
    
    # Pré-allouer les tableaux
    edge_rows = np.zeros(n_edges, dtype=np.int32)
    edge_cols = np.zeros(n_edges, dtype=np.int32)
    edge_values = np.ones(n_edges, dtype=np.float32)  # Poids uniformes
    
    # Construire les arêtes (chaque nœud a 'degree' voisins)
    for i in range(n_nodes):
        for k in range(degree):
            idx = i * degree + k
            edge_rows[idx] = i
            # Choisir un voisin aléatoire différent de i
            j = rng.integers(0, n_nodes)
            while j == i:
                j = rng.integers(0, n_nodes)
            edge_cols[idx] = j
    
    # Générer les fréquences
    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0,
        gamma=1.0,
        symmetric=False,
        seed=123,
    )
    omega = frequency_generator.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)
    
    # Créer le modèle sparse
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
    header = ["N", "degree", "n_edges", "mean_time", "std_time", "min_time", "max_time", "n_measures"]
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
    n_measures = 3  # 3 mesures pour réduire le temps

    # Tailles à tester
    N_values = [500, 1000, 5000, 10000, 20000, 30000, 40000, 100000,300000,1000000 ]

    # Degré fixe (nombre de voisins par nœud)
    DEGREE = 10  # ← Ajustez selon vos besoins (10, 20, 50, etc.)


    print("TEST DE PERFORMANCE : CAS SPARSE (CSR)")
   
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}")
    print(f"Nombre de mesures par N = {n_measures}")
    print(f"Degré par nœud = {DEGREE}")
    print(f"Nombre total d'arêtes ≈ N × {DEGREE}")
  

    results = []

    for N in N_values:
        n_edges = N * DEGREE
        print(f"\n N = {N} (arêtes = {n_edges})")
        

        try:
            model, theta0 = build_sparse_problem(N, degree=DEGREE)
        except Exception as e:
            print(f"  Erreur lors de la construction : {e}")
            results.append([N, DEGREE, n_edges, None, None, None, None, 0])
            continue
        
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
            
            print(f"\n   statistiques :")
            print(f"    Moyenne  : {mean_time:.6f}s")
            print(f"    Écart-type : {std_time:.6f}s")
            print(f"    Min      : {min_time:.6f}s")
            print(f"    Max      : {max_time:.6f}s")
            
            results.append([N, DEGREE, n_edges, mean_time, std_time, min_time, max_time, len(valid_times)])
        else:
            print(f"  Aucune mesure valide pour N={N}")
            results.append([N, DEGREE, n_edges, None, None, None, None, 0])

    # Sauvegarder les résultats
    save_results_to_csv(results)

    # Résumé
   
    print("RÉSUMÉ DES PERFORMANCES (CAS SPARSE)")
  
    print(f"{'N':<12} {'Arêtes':<14} {'Moyenne (s)':<16} {'Écart-type':<16} {'Mesures':<8}")
  
    for row in results:
        N, deg, n_edges, mean_t, std_t, _, _, n_meas = row
        if mean_t is not None:
            print(f"{N:<12} {n_edges:<14} {mean_t:<16.6f} ±{std_t:<15.6f} {n_meas:<8}")
        else:
            print(f"{N:<12} {n_edges:<14} {'---':<16} {'---':<16} {n_meas:<8}")

   
    print("Test terminé")
    print(f"  - Modèle : Sparse (CSR)")
    print(f"  - Degré moyen : {DEGREE}")
    print(f"  - Complexité théorique : O(N × {DEGREE})")
  


if __name__ == "__main__":
    main()