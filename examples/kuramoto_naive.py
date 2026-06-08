"""Benchmark and consistency check for the naive Kuramoto implementations."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from neuromass.models.kuramoto import NaiveKuramotoModel
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
    n_nodes = 100
    adjacency = rng.random((n_nodes, n_nodes))
    adjacency *= rng.random((n_nodes, n_nodes)) < 0.1
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


def main() -> None:
    """Run the naive Kuramoto backend benchmark and visualization demo.

    The script compares the Python, Cython, C, and C++ implementations on the
    same small random problem, reports execution times and numerical errors,
    and displays representative trajectories, the order parameter, and a phase
    colormap for all nodes.
    """

    model, theta0 = build_demo_problem()
    T = 50.0
    dt = 0.01

    backends = ["python", "cython", "c", "cpp"]
    results: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for backend in backends:
        start = perf_counter()
        time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
        elapsed = perf_counter() - start
        results[backend] = (time, theta, elapsed)

    reference_theta = results["python"][1]

    print("Naive Kuramoto benchmark")
    print(f"Available backends: {', '.join(model.available_backends())}")
    print(f"{'backend':<10} {'time [s]':>12} {'max abs err':>14}")
    for backend in backends:
        _, theta, elapsed = results[backend]
        error = np.max(np.abs(theta - reference_theta))
        print(f"{backend:<10} {elapsed:12.6f} {error:14.6e}")

    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    for node_idx in range(5):
        axes[0].plot(
            results["python"][0],
            reference_theta[node_idx],
            label=f"node {node_idx}",
        )
    axes[0].set_title("Reference Python trajectories for 5 oscillators")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend(loc="upper right")

    for backend in backends:
        time, theta, _ = results[backend]
        axes[1].plot(time, order_parameter(theta), label=backend)
    axes[1].set_title("Kuramoto order parameter across implementations")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$|R(t)|$")
    axes[1].legend(loc="best")

    phase_map = axes[2].imshow(
        wrap_phase(reference_theta),
        aspect="auto",
        origin="lower",
        extent=(results["python"][0][0], results["python"][0][-1], 0, model.n_nodes - 1),
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[2].set_title("Wrapped phase map for all oscillators")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("node index")
    figure.colorbar(phase_map, ax=axes[2], label=r"wrapped $\theta$")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
