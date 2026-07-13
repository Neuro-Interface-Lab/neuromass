"""Tests for the sparse Kuramoto solver implementations."""

from __future__ import annotations

import numpy as np
import pytest


def build_sparse_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    rng = np.random.default_rng(42)
    n_nodes = 10
    adjacency = rng.random((n_nodes, n_nodes))
    adjacency *= rng.random((n_nodes, n_nodes)) < 0.3
    np.fill_diagonal(adjacency, 0.0)
    adjacency = (adjacency + adjacency.T) / 2.0

    edge_rows, edge_cols = np.nonzero(adjacency)
    edge_values = adjacency[edge_rows, edge_cols]
    n_edges = edge_values.shape[0]

    omega = rng.normal(scale=1.0, size=n_nodes)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    return omega, theta0, edge_values.astype(np.float64), edge_rows.astype(np.intc), edge_cols.astype(np.intc), n_edges


def python_sparse_kernel(
    edge_values: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    epsilon: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    n_nodes = omega.shape[0]
    theta = np.zeros((n_nodes, n_steps + 1), dtype=np.float64)
    theta[:, 0] = theta0

    for step in range(n_steps):
        coupling = np.zeros(n_nodes, dtype=np.float64)
        for e in range(edge_values.shape[0]):
            i = edge_rows[e]
            j = edge_cols[e]
            coupling[i] += edge_values[e] * np.sin(theta[j, step] - theta[i, step])
        theta[:, step + 1] = theta[:, step] + dt * (
            omega + (epsilon / n_nodes) * coupling
        )

    return theta


def get_sparse_backends() -> dict[str, object]:
    backends: dict[str, object] = {}

    try:
        import neuromass.models.kuramoto._native.c_backend as c_backend

        backends["c"] = c_backend
    except ImportError:
        pass

    try:
        import neuromass.models.kuramoto._native.cpp_backend as cpp_backend

        backends["cpp"] = cpp_backend
    except ImportError:
        pass

    return backends


def test_simu_sparse_python_reference_shape() -> None:
    omega, theta0, edge_values, edge_rows, edge_cols, n_edges = build_sparse_problem()
    epsilon = 1.5
    T = 1.0
    dt = 0.1
    n_steps = int(round(T / dt))

    theta = python_sparse_kernel(
        edge_values,
        edge_rows,
        edge_cols,
        omega,
        theta0,
        epsilon,
        dt,
        n_steps,
    )

    assert theta.shape == (omega.shape[0], n_steps + 1)
    assert np.all(np.isfinite(theta))


def test_simu_sparse_backends_match_python_reference() -> None:
    omega, theta0, edge_values, edge_rows, edge_cols, n_edges = build_sparse_problem()
    epsilon = 1.5
    T = 1.0
    dt = 0.1
    n_steps = int(round(T / dt))

    reference = python_sparse_kernel(
        edge_values,
        edge_rows,
        edge_cols,
        omega,
        theta0,
        epsilon,
        dt,
        n_steps,
    )

    backends = get_sparse_backends()
    if not backends:
        pytest.skip("No compiled sparse backends are available.")

    for name, module in backends.items():
        candidate = module.simu_sparse(
            edge_values,
            edge_rows,
            edge_cols,
            omega,
            theta0,
            epsilon,
            dt,
            n_steps,
            n_edges,
            edge_rows,
            edge_cols,
        )
        assert candidate.shape == reference.shape
        assert np.allclose(candidate, reference, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    test_simu_sparse_python_reference_shape()
    test_simu_sparse_backends_match_python_reference()
    print("simu_sparse smoke tests passed.")
