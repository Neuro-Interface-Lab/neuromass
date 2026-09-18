import csv
import time
import numpy as np
import pynvml
 
from neuromass.models.kuramoto import SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator
 
results_csv = "execution_times_GPU_sparse_stats.csv"
memory_csv = "gpu_memory_usage_sparse.csv"
 
 

pynvml.nvmlInit()
GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0 ; change l'index si besoin
 
 
def get_gpu_memory_used_mb():
    """Mémoire GPU actuellement utilisée (en MB), mesurée via NVML."""
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        return info.used / (1024 ** 2)  # bytes -> MB
    except AttributeError:
        return None
    except Exception as e:
        # pynvml raises NVMLError_NotSupported on some platforms/drivers;
        # return None so callers can handle 'not available' gracefully.
        try:
            if e.__class__.__name__ == 'NVMLError_NotSupported' or 'Not Supported' in str(e):
                return None
        except Exception:
            pass
        print(f"Warning: NVML error retrieving GPU memory: {e}")
        return None
 
 
def build_sparse_problem(n_nodes, degree=10):
    """
    Construire un problème Kuramoto sparse avec un degré fixe.
    Évite la construction de la matrice dense N×N.
    """
    rng = np.random.default_rng(42)
 
    n_edges = n_nodes * degree
 
    edge_rows = np.zeros(n_edges, dtype=np.int32)
    edge_cols = np.zeros(n_edges, dtype=np.int32)
    edge_values = np.ones(n_edges, dtype=np.float32)  # Poids uniformes
 
    for i in range(n_nodes):
        for k in range(degree):
            idx = i * degree + k
            edge_rows[idx] = i
            j = rng.integers(0, n_nodes)
            while j == i:
                j = rng.integers(0, n_nodes)
            edge_cols[idx] = j
 
    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0, gamma=1.0, symmetric=False, seed=123,
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
 
 
def execution_time_and_memory(model, theta0, T, dt, backend):
    """Mesure le temps d'exécution ET le pic de mémoire GPU utilisée."""
    mem_before = get_gpu_memory_used_mb()
 
    start = time.perf_counter()
    time_arr, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    elapsed = time.perf_counter() - start
 
    mem_after = get_gpu_memory_used_mb()
    if mem_before is None or mem_after is None:
        mem_delta = None
    else:
        mem_delta = mem_after - mem_before  # approx. mémoire consommée par cet appel
 
    return elapsed, mem_before, mem_after, mem_delta
 
 
def save_results_to_csv(results, filename=results_csv):
    header = ["N", "degree", "n_edges", "mean_time", "std_time", "min_time", "max_time", "n_measures"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)
    print(f"\nRésultats (temps) sauvegardés dans : {filename}")
 
 
def save_memory_to_csv(memory_results, filename=memory_csv):
    header = ["N", "degree", "n_edges", "mem_before_mb", "mem_after_mb", "mem_delta_mb"]
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in memory_results:
            writer.writerow(row)
    print(f"Résultats (mémoire GPU) sauvegardés dans : {filename}")
 
 
def main():
    T = 0.5
    dt = 0.01
    n_steps = int(T / dt)
    n_measures = 3
 
    N_values = [500, 1000, 5000, 10000, 20000, 30000, 40000, 100000, 300000, 1000000]
    DEGREE = 10
 
    print("TEST DE PERFORMANCE + MEMOIRE GPU : CAS SPARSE (CSR)")
    print(f"dt = {dt}s, T = {T}s, steps = {n_steps}")
    print(f"Nombre de mesures par N = {n_measures}")
    print(f"Degré par nœud = {DEGREE}")
 
    results = []
    memory_results = []
 
    for N in N_values:
        n_edges = N * DEGREE
        print(f"\n N = {N} (arêtes = {n_edges})")
 
        try:
            model, theta0 = build_sparse_problem(N, degree=DEGREE)
        except Exception as e:
            print(f"  Erreur lors de la construction : {e}")
            results.append([N, DEGREE, n_edges, None, None, None, None, 0])
            memory_results.append([N, DEGREE, n_edges, None, None, None])
            continue
 
        times = []
        mem_deltas = []
 
        for m in range(n_measures):
            try:
                elapsed, mem_before, mem_after, mem_delta = execution_time_and_memory(
                    model, theta0, T, dt, backend="c"
                )
                times.append(elapsed)
                mem_deltas.append(mem_delta)
                if mem_before is None or mem_after is None or mem_delta is None:
                    print(f"  Mesure {m+1}/{n_measures} : {elapsed:.6f}s | GPU mem: Not available")
                else:
                    print(f"  Mesure {m+1}/{n_measures} : {elapsed:.6f}s | "
                          f"GPU mem: {mem_before:.1f} -> {mem_after:.1f} MB (Δ={mem_delta:.1f} MB)")
            except Exception as e:
                print(f"  Mesure {m+1}/{n_measures} : ERREUR - {e}")
 
        valid_times = [t for t in times if t is not None]
 
        if valid_times:
            mean_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            results.append([N, DEGREE, n_edges, mean_time, std_time, min_time, max_time, len(valid_times)])
        else:
            results.append([N, DEGREE, n_edges, None, None, None, None, 0])
 
        # Filtrer les mesures valides (non None) avant d'utiliser numpy
        mem_valid = [m for m in mem_deltas if m is not None]
        if mem_valid:
            # On garde le pic (max) de mémoire consommée observé sur les mesures valides
            mem_peak = max(mem_valid)
            # NVML peut être indisponible; on n'a pas forcément mem_before/mem_after valides
            memory_results.append([N, DEGREE, n_edges, None, None, mem_peak])
        else:
            memory_results.append([N, DEGREE, n_edges, None, None, None])
 
    save_results_to_csv(results)
    save_memory_to_csv(memory_results)
 
    print("\nTest terminé.")
 
 
if __name__ == "__main__":
    main()
 