"""Benchmark and consistency check for the naive Kuramoto implementations."""

from time import perf_counter
import csv
import os
import numpy as np

from neuromass.models.kuramoto import NaiveKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


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

def execution_time(model, theta0, T, dt, backend):
    """Measure the execution time of the Kuramoto model simulation."""
 
    start = perf_counter()
    time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
    elapsed = perf_counter() - start
    return elapsed

def save_results_to_csv(n_nodes, results, filename="execution_times_parallelism.csv"):
    """Save the execution times to a CSV file."""
    file_exists = os.path.isfile(filename)
    header = ["N", "backend", "mean_time"]
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        for  backend, elapsed in results.items():
            writer.writerow([n_nodes, backend, elapsed])

def main():
    n_nodes =500
    dt = 0.01
    T = 50.0

    backends = ["python", "cython", "c", "cpp"]
    

    model, theta0 = build_problem(n_nodes )
    results = {}
    for backend in backends:
        try:
            elapsed = execution_time(model, theta0, T, dt, backend)
            results[backend] = elapsed
            print(f"{backend:<10} : {elapsed:.6f}s")
        except Exception as e:
            results[backend] = None
            print(f"{backend}:<10 : erreur - {e}")
    save_results_to_csv(n_nodes, results,"execution_times_parallelism.csv")

if __name__ == "__main__":
    main()