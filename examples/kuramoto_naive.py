"""Benchmark and consistency check for the naive Kuramoto implementations."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import coo_matrix
from neuromass.models.kuramoto import NaiveKuramotoModel, MeanFieldKuramotoModel,SparseKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


def build_demo_problem() -> tuple[NaiveKuramotoModel, np.ndarray]:
    """Build a small Kuramoto benchmark problem.

    Returns
    -------
    tuple[NaiveKuramotoModel, numpy.ndarray]
        A tuple containing the configured Kuramoto model and the initial
        phase vector of shape ``(n_nodes,)``.
    """

    rng = np.random.default_rng(42)
    n_nodes = 200
    epsilon = 3.0
    adjacency = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adjacency, 0.0)

    from scipy.sparse import coo_matrix
    sparse_adjacency = coo_matrix(adjacency)
    edge_rows = sparse_adjacency.row.astype(np.int32)
    edge_cols = sparse_adjacency.col.astype(np.int32)
    edge_values = sparse_adjacency.data.astype(np.float64)


    frequency_generator = LorentzianFrequencyGenerator(
        x0=0.0,
        gamma=1.0,
        symmetric=False,
        seed=123,
    )
    omega = frequency_generator.sample(n_nodes, truncated=True, cutoff=5.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)
    

    model_naive = NaiveKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=epsilon,
        adjacency=adjacency,
    )
    model_meanfield = MeanFieldKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=epsilon,
    )
    model_sparse = SparseKuramotoModel(
        n_nodes=n_nodes,
        n_edges=len(edge_values),
        edge_values=edge_values,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        omega=omega,
        epsilon=epsilon,
    )
    return { 
    "naive":model_naive, 
    "order_parameter":model_meanfield,
    "sparse":  model_sparse, 
    "theta0": theta0,
    "n_nodes": n_nodes,
    "n_edges": len(edge_values),
    }



def order_parameter(theta: np.ndarray) -> np.ndarray:
    """Compute the Kuramoto order parameter over time.

    Parameters
    ----------
    theta : numpy.ndarray
        Phase trajectories with shape ``(n_nodes, n_times)``.

    Returns
    -------
    numpy.ndarray
        Time series of the order parameter with shape ``(n_times,)``.
    """

    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def wrap_phase(theta: np.ndarray) -> np.ndarray:
    """Wrap phases to the interval ``[-pi, pi]``.

    Parameters
    ----------
    theta : numpy.ndarray
        Phase trajectories with shape ``(n_nodes, n_times)``.

    Returns
    -------
    numpy.ndarray
        Wrapped phases with the same shape as ``theta``.
    """

    return np.angle(np.exp(1j * theta))

def run_benchmark(model, theta0, T, dt, model_name):
    """Run benchmark for a given model."""
    
    backends = ["python", "cython", "c", "cpp"]
    results = {}
    
    print(f"\n{model_name} model")
    print("-" * 60)
    
    for backend in backends:
        try:
            start = perf_counter()
            time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
            elapsed = perf_counter() - start
            results[backend] = (time, theta, elapsed)
            print(f"  {backend:<10} : {elapsed:8.4f} s")
        except Exception as e:
            print(f"  {backend:<10} : ERREUR - {e}")
            results[backend] = (None, None, None)
    
    return results, backends


def plot_results(results_naive, results_order_parameter, results_sparse, time, n_nodes):
    """Plot the results for all three models."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    models = [
        ("naive", results_naive),
        ("order_parameter", results_order_parameter),
        ("sparse", results_sparse),
    ]
    
    for col, (model_name, results) in enumerate(models):
        # Ligne 1 : Trajectoires
        ax = axes[0, col]
        if results["python"][1] is not None:
            theta_python = results["python"][1]
            for node_idx in range(5):
                ax.plot(time, theta_python[node_idx], label=f"node {node_idx}")
        ax.set_title(f"{model_name} - Trajectoires ")
        ax.set_ylabel(r"$\theta$")
        ax.legend(loc="upper right")
        
        # Ligne 2 : ParamÃ¨tre d'ordre
        ax = axes[1, col]
        backends = ["python", "cython", "c", "cpp"]
        for backend in backends:
            if results[backend][1] is not None:
                _, theta, _ = results[backend]
                label = "Python" if backend == "python" else backend
                linestyle = '-' if backend == "python" else '--'
                linewidth = 2 if backend == "python" else 1.5
                ax.plot(time, order_parameter(theta), label=label.capitalize(),
                       linestyle=linestyle, linewidth=linewidth)
        ax.set_title(f"{model_name} - order R(t)")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$|R(t)|$")
        ax.legend(loc="best")
        
        # Ligne 3 : Carte des phases
        ax = axes[2, col]
        if results["python"][1] is not None:
            phase_map = ax.imshow(
                wrap_phase(results["python"][1]),
                aspect="auto",
                origin="lower",
                extent=(time[0], time[-1], 0, n_nodes - 1),
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
            )
            plt.colorbar(phase_map, ax=ax, label=r"wrapped $\theta$")
        ax.set_title(f"{model_name} - Carte des phases")
        ax.set_xlabel("time")
        ax.set_ylabel("node index")
    
    fig.tight_layout()
    plt.show()


def print_model_summary(model_name: str, results: dict[str, tuple[object, object, float]]) -> None:
    """Print a small benchmark summary for a model."""

    reference = results["python"][1]
    print(f"\n{model_name}")
    print(f"{'backend':<10} {'time (s)':>12}{'max abs err':>14}")
    print("-" * 36)

    for backend in ["python", "cython", "c", "cpp"]:
        time_value = results[backend][2]
        theta = results[backend][1]

        if theta is None or reference is None:
            time_str = f"{'N/A':>12}"
            err_str = f"{'N/A':>14}"
        else:
            err = 0.0 if backend == "python" else np.max(np.abs(theta - reference))
            time_str = f"{time_value:>12.4f}"
            err_str = f"{err:>14.6e}"

        print(f"{backend:<10}{time_str}{err_str}")


def print_performance_summary(results_naive, results_order_parameter, results_sparse):
    """Print performance summary for all models."""
    
    print("\n" + "=" * 80)
    print(" PERFORMANCES (temps en secondes)")
    print("=" * 80)
    
    backends = ["python", "cython", "c", "cpp"]
    
    # En-tÃªte
    print(f"{'backend':<10} {'naive':>12} {'order_parameter':>12} {'sparse':>12}")
    print("-" * 50)
    
    # Lignes par backend
    for backend in backends:
        t_naive = results_naive.get(backend, (None, None, None))[2]
        t_order_parameter = results_order_parameter.get(backend, (None, None, None))[2]
        t_sparse = results_sparse.get(backend, (None, None, None))[2]
        
        naive_str = f"{t_naive:.4f}" if t_naive is not None else "N/A"
        order_parameter_str = f"{t_order_parameter:.4f}" if t_order_parameter is not None else "N/A"
        sparse_str = f"{t_sparse:.4f}" if t_sparse is not None else "N/A"
        
        print(f"{backend:<10} {naive_str:>12} {order_parameter_str:>12} {sparse_str:>12}")
    
    # AccÃ©lÃ©rations
    print("\n" + "-" * 50)
    print("(C++ / Python)")
    print("-" * 50)
    
    if results_naive["python"][2] and results_naive["cpp"][2]:
        speedup = results_naive["python"][2] / results_naive["cpp"][2]
        print(f"  Dense  : {speedup:.2f}x")
    if results_order_parameter["python"][2] and results_order_parameter["cpp"][2]:
        speedup = results_order_parameter["python"][2] / results_order_parameter["cpp"][2]
        print(f"  Global : {speedup:.2f}x")
    if results_sparse["python"][2] and results_sparse["cpp"][2]:
        speedup = results_sparse["python"][2] / results_sparse["cpp"][2]
        print(f"  Sparse : {speedup:.2f}x")



def main() -> None:
    """Run the Kuramoto backend benchmark for all implementations."""
    
    print("=" * 70)
    print("KURAMOTO BENCHMARK - naive / order_parameter / SPARSE")
    print("=" * 70)

    problem = build_demo_problem()
    naive_model = problem["naive"]
    order_parameter_model = problem["order_parameter"]
    sparse_model = problem["sparse"]
    theta0 = problem["theta0"]
    n_nodes = problem["n_nodes"]

    T = 10.0
    dt = 0.01


    results_naive, backends = run_benchmark(naive_model, theta0, T, dt, "Model (avec adjacency)")
    print_model_summary("Model (avec adjacency)", results_naive)

    results_order_parameter, _ = run_benchmark(order_parameter_model, theta0, T, dt, "Model order parameter (mean-field)")
    print_model_summary("Model order parameter (mean-field)", results_order_parameter)

    results_sparse, _ = run_benchmark(sparse_model, theta0, T, dt, "Model sparse (COO)")
    print_model_summary("Model sparse (COO)", results_sparse)
    
    time = results_naive["python"][0]
    plot_results(results_naive, results_order_parameter, results_sparse, time, n_nodes)


if __name__ == "__main__":
    main()