import numpy as np

from neuromass.models.kuramoto import NaiveKuramotoModel


def test_python_backend_shape() -> None:
    n_nodes = 4
    omega = np.linspace(0.1, 0.4, n_nodes)
    adjacency = np.ones((n_nodes, n_nodes), dtype=np.float64) - np.eye(n_nodes)
    theta0 = np.zeros(n_nodes, dtype=np.float64)

    model = NaiveKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=0.5,
        adjacency=adjacency,
    )

    time, theta = model.solve(theta0=theta0, T=1.0, dt=0.1, backend="python")

    assert time.shape == (11,)
    assert theta.shape == (n_nodes, 11)


def test_compiled_backends_match_python_reference() -> None:
    n_nodes = 5
    rng = np.random.default_rng(123)
    omega = rng.normal(size=n_nodes)
    adjacency = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(adjacency, 0.0)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_nodes)

    model = NaiveKuramotoModel(
        n_nodes=n_nodes,
        omega=omega,
        epsilon=1.2,
        adjacency=adjacency,
    )

    _, reference = model.solve(theta0=theta0, T=0.4, dt=0.1, backend="python")

    for backend in ("cython", "c", "cpp"):
        if backend not in model.available_backends():
            continue
        _, candidate = model.solve(theta0=theta0, T=0.4, dt=0.1, backend=backend)
        assert np.allclose(candidate, reference)

