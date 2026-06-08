"""High-level Python interface for Kuramoto models."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import BaseModel

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
BackendKernel = Callable[[FloatMatrix, FloatVector, FloatVector, float, float, int], FloatMatrix]

_BACKEND_MODULES = {
    "cython": "neuromass.models.kuramoto._native.cython_backend",
    "c": "neuromass.models.kuramoto._native.c_backend",
    "cpp": "neuromass.models.kuramoto._native.cpp_backend",
}


def _python_kernel(
    adjacency: FloatMatrix,
    omega: FloatVector,
    theta0: FloatVector,
    epsilon: float,
    dt: float,
    n_steps: int,
) -> FloatMatrix:
    """Reference NumPy/Python implementation of the naive Kuramoto solver."""

    n_nodes = omega.shape[0]
    theta = np.zeros((n_nodes, n_steps + 1), dtype=np.float64)
    theta[:, 0] = theta0

    for step in range(n_steps):
        for i in range(n_nodes):
            coupling = 0.0
            for j in range(n_nodes):
                weight = adjacency[i, j]
                if weight != 0.0:
                    coupling += weight * np.sin(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega[i] + (epsilon / n_nodes) * coupling
            )

    return theta


def _load_backend(backend: str) -> BackendKernel:
    """Resolve a simulation backend by name."""

    if backend == "python":
        return _python_kernel

    if backend not in _BACKEND_MODULES:
        available = ", ".join(sorted(["python", *_BACKEND_MODULES.keys()]))
        raise ValueError(f"Unknown backend '{backend}'. Available backends: {available}.")

    module_name = _BACKEND_MODULES[backend]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Backend '{backend}' is not available. Rebuild the package with "
            "`pip install -e . --no-build-isolation` in the `neuromass` environment."
        ) from exc

    return module.simulate_naive_kuramoto


@dataclass(slots=True)
class NaiveKuramotoModel(BaseModel):
    """Generalist naive Kuramoto model with interchangeable solver backends."""

    n_nodes: int
    omega: ArrayLike
    epsilon: float
    adjacency: ArrayLike
    name: str = field(default="kuramoto-naive", init=False)

    def __post_init__(self) -> None:
        self.omega = np.ascontiguousarray(self.omega, dtype=np.float64)
        self.adjacency = np.ascontiguousarray(self.adjacency, dtype=np.float64)

        if self.n_nodes <= 0:
            raise ValueError("`n_nodes` must be strictly positive.")

        if self.omega.shape != (self.n_nodes,):
            raise ValueError(
                "`omega` must be a one-dimensional vector of shape "
                f"({self.n_nodes},)."
            )

        if self.adjacency.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                "`adjacency` must be a square matrix of shape "
                f"({self.n_nodes}, {self.n_nodes})."
            )

    @staticmethod
    def available_backends() -> list[str]:
        """List the backends currently importable in this environment."""

        available = ["python"]
        for backend, module_name in _BACKEND_MODULES.items():
            try:
                import_module(module_name)
            except ImportError:
                continue
            available.append(backend)
        return available

    def solve(
        self,
        theta0: ArrayLike,
        T: float,
        dt: float,
        backend: str = "python",
    ) -> tuple[FloatVector, FloatMatrix]:
        """Solve the Kuramoto system with an explicit Euler scheme."""

        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(
                "`theta0` must be a one-dimensional vector of shape "
                f"({self.n_nodes},)."
            )

        if T <= 0.0:
            raise ValueError("`T` must be strictly positive.")
        if dt <= 0.0:
            raise ValueError("`dt` must be strictly positive.")

        n_steps_float = T / dt
        n_steps = int(round(n_steps_float))
        if not np.isclose(n_steps_float, n_steps):
            raise ValueError("`T / dt` must be an integer so the time grid is well defined.")

        kernel = _load_backend(backend)
        theta = kernel(
            self.adjacency,
            self.omega,
            theta0_array,
            float(self.epsilon),
            float(dt),
            n_steps,
        )
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta
