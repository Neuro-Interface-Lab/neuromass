"""Finite-size Kuramoto sanity check against the Lorentzian critical coupling."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from neuromass.models.kuramoto import NaiveKuramotoModel
from neuromass.utils import LorentzianFrequencyGenerator


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


def theoretical_order_parameter(epsilon: np.ndarray, epsilon_c: float) -> np.ndarray:
    """Evaluate the infinite-size Lorentzian mean-field prediction.

    Parameters
    ----------
    epsilon : numpy.ndarray
        Coupling values at which the theoretical asymptotic order parameter is
        evaluated.
    epsilon_c : float
        Critical coupling predicted by theory.

    Returns
    -------
    numpy.ndarray
        Theoretical asymptotic order parameter evaluated at ``epsilon``.
    """

    ratio = np.clip(1.0 - (epsilon_c / epsilon), a_min=0.0, a_max=None)
    return np.sqrt(ratio)


def build_problem(n_nodes: int, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a finite-size Kuramoto problem for the critical-coupling sweep.

    Parameters
    ----------
    n_nodes : int
        Number of oscillators in the all-to-all network.
    gamma : float
        Half-width parameter of the Lorentzian frequency distribution.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Natural frequencies, adjacency matrix, and initial phase vector.
    """

    generator = LorentzianFrequencyGenerator(x0=0.0, gamma=gamma, symmetric=True)
    omega = generator.sample(
        n_nodes,
        truncated=True,
        cutoff=20.0 * gamma,
        method="quantile",
    )
    adjacency = np.ones((n_nodes, n_nodes), dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    theta0 = np.linspace(-np.pi, np.pi, n_nodes, endpoint=False, dtype=np.float64)
    return omega, adjacency, theta0


def main() -> None:
    """Run a finite-size sweep to compare ``r_inf`` with the theoretical onset.

    The script scans logarithmically spaced coupling values, estimates the
    asymptotic order parameter from the tail of each simulation, and compares
    the numerical transition with the theoretical critical coupling of the
    Lorentzian Kuramoto model.
    """

    n_nodes = 100
    gamma = 1.0
    dt = 0.01
    T = 200.0
    tail_fraction = 0.25
    backend = "cpp"

    epsilon_values = np.logspace(-1, 1, 20)
    epsilon_c = 2.0 * gamma

    omega, adjacency, theta0 = build_problem(n_nodes=n_nodes, gamma=gamma)
    r_infinity = np.zeros_like(epsilon_values)
    compute_times = np.zeros_like(epsilon_values)

    print("Kuramoto critical-coupling sweep")
    print(f"N = {n_nodes}, gamma = {gamma:.3f}, predicted epsilon_c = {epsilon_c:.3f}")
    print(f"backend = {backend}, T = {T:.1f}, dt = {dt:.3f}")
    print(f"{'epsilon':>10} {'r_inf':>12} {'time [s]':>12}")

    for idx, epsilon in enumerate(epsilon_values):
        model = NaiveKuramotoModel(
            n_nodes=n_nodes,
            omega=omega,
            epsilon=float(epsilon),
            adjacency=adjacency,
        )

        start = perf_counter()
        time, theta = model.solve(theta0=theta0, T=T, dt=dt, backend=backend)
        compute_times[idx] = perf_counter() - start

        r_t = order_parameter(theta)
        tail_start = int((1.0 - tail_fraction) * r_t.shape[0])
        r_infinity[idx] = np.mean(r_t[tail_start:])
        print(f"{epsilon:10.4f} {r_infinity[idx]:12.6f} {compute_times[idx]:12.6f}")

    theory = theoretical_order_parameter(epsilon_values, epsilon_c)
    onset_threshold = 0.2
    onset_indices = np.flatnonzero(r_infinity > onset_threshold)
    numerical_epsilon_c = (
        float(epsilon_values[onset_indices[0]])
        if onset_indices.size > 0
        else float("nan")
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)

    axes[0].semilogx(epsilon_values, r_infinity, "o-", label=r"numerical $r_\infty$")
    axes[0].semilogx(epsilon_values, theory, "--", label="Lorentzian mean-field theory")
    axes[0].axvline(epsilon_c, color="black", linestyle=":", label=rf"predicted $\epsilon_c = {epsilon_c:.2f}$")
    axes[0].axvline(
        numerical_epsilon_c,
        color="tab:red",
        linestyle=":",
        label=rf"finite-size onset $\approx {numerical_epsilon_c:.2f}$",
    )
    axes[0].set_title("Asymptotic Kuramoto order parameter vs coupling")
    axes[0].set_ylabel(r"$r_\infty$")
    axes[0].legend(loc="best")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(epsilon_values, compute_times, "o-", color="tab:green")
    axes[1].axvline(epsilon_c, color="black", linestyle=":")
    axes[1].set_title("Computation time along the coupling sweep")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel("time [s]")
    axes[1].grid(True, which="both", alpha=0.3)

    print()
    print(f"Predicted epsilon_c = {epsilon_c:.6f}")
    print(
        f"Observed finite-size onset (threshold r_inf > {onset_threshold:.1f}) = "
        f"{numerical_epsilon_c:.6f}"
    )

    plt.show()


if __name__ == "__main__":
    main()
