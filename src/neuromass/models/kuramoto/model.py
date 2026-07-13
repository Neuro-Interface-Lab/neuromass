"""High-level Python interface for Kuramoto models."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import BaseModel

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


_BACKEND_MODULES = {
    "cython": "neuromass.models.kuramoto._native.cython_backend",
    "c": "neuromass.models.kuramoto._native.c_backend",
    "cpp": "neuromass.models.kuramoto._native.cpp_backend",
}

def _python_kernel_naive(
    adjacency: FloatMatrix,
    omega: FloatVector,
    theta0: FloatVector,
    epsilon: float,
    dt: float,
    n_steps: int,
) -> FloatMatrix:
    """Reference Python implementation (dense network)."""
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


def _python_kernel_order_parameter(
    omega: FloatVector,
    theta0: FloatVector,
    epsilon: float,
    dt: float,
    n_steps: int,
) -> FloatMatrix:
    """Reference Python implementation (mean-field global coupling)."""
    n_nodes = omega.shape[0]
    theta = np.zeros((n_nodes, n_steps + 1), dtype=np.float64)
    theta[:, 0] = theta0

    for step in range(n_steps):
        c = np.mean(np.cos(theta[:, step]))
        s = np.mean(np.sin(theta[:, step]))
        r = np.sqrt(c * c + s * s)
        psi = np.arctan2(s, c)
        theta[:, step + 1] = theta[:, step] + dt * (
            omega + epsilon * r * np.sin(psi - theta[:, step])
        )
    return theta


def _python_kernel_sparse(
    edge_values: NDArray,
    edge_rows: NDArray,
    edge_cols: NDArray,
    omega: FloatVector,
    theta0: FloatVector,
    epsilon: float,
    dt: float,
    n_steps: int,
) -> FloatMatrix:
    """Reference Python implementation (sparse COO)."""
    n_nodes = omega.shape[0]
    n_edges = edge_values.shape[0]
    theta = np.zeros((n_nodes, n_steps + 1), dtype=np.float64)
    theta[:, 0] = theta0

    for step in range(n_steps):
        coupling = np.zeros(n_nodes)
        for e in range(n_edges):
            i = edge_rows[e]
            j = edge_cols[e]
            weight = edge_values[e]
            coupling[i] += weight * np.sin(theta[j, step] - theta[i, step])
        theta[:, step + 1] = theta[:, step] + dt * (
            omega + (epsilon / n_nodes) * coupling
        )
    return theta


# ============================================================
# Chargeurs de backends
# ============================================================

def _load_backend_naive(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_naive
    if backend not in _BACKEND_MODULES:
        raise ValueError(f"Unknown backend '{backend}'.")
    module = import_module(_BACKEND_MODULES[backend])
    return module.simulate_naive_kuramoto


def _load_backend_order_parameter(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_order_parameter
    if backend not in _BACKEND_MODULES:
        raise ValueError(f"Unknown backend '{backend}'.")
    module = import_module(_BACKEND_MODULES[backend])
    return module.simu_para_complexe


_load_backend_global = _load_backend_order_parameter


def _load_backend_sparse(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_sparse
    if backend not in _BACKEND_MODULES:
        raise ValueError(f"Unknown backend '{backend}'.")
    module = import_module(_BACKEND_MODULES[backend])

    def _sparse_wrapper(
        edge_values,
        edge_rows,
        edge_cols,
        omega,
        theta0,
        epsilon,
        dt,
        n_steps,
    ):
        if backend == "cython":
            return module.simu_sparse(
                omega,
                theta0,
                edge_values,
                edge_rows,
                edge_cols,
                epsilon,
                dt,
                n_steps,
            )

        n_edges = int(edge_values.shape[0])
        return module.simu_sparse(
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

    return _sparse_wrapper



# Classe 1 : model naive 

@dataclass(slots=True)
class NaiveKuramotoModel(BaseModel):
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
            raise ValueError(f"`omega` must be of shape ({self.n_nodes},).")
        if self.adjacency.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(f"`adjacency` must be of shape ({self.n_nodes}, {self.n_nodes}).")

    @staticmethod
    def available_backends() -> list[str]:
        available = ["python"]
        for backend, module_name in _BACKEND_MODULES.items():
            try:
                import_module(module_name)
            except ImportError:
                continue
            available.append(backend)
        return available

    def solve(self, theta0: ArrayLike, T: float, dt: float, backend: str = "python") -> tuple[FloatVector, FloatMatrix]:
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_naive(backend)
        theta = kernel(self.adjacency, self.omega, theta0_array, float(self.epsilon), float(dt), n_steps)
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta



# Classe 2 : model order parameter (mean-field)

@dataclass(slots=True)
class MeanFieldKuramotoModel(BaseModel):
    n_nodes: int
    omega: ArrayLike
    epsilon: float
    name: str = field(default="kuramoto-meanfield", init=False)

    def __post_init__(self) -> None:
        self.omega = np.ascontiguousarray(self.omega, dtype=np.float64)
        if self.n_nodes <= 0:
            raise ValueError("`n_nodes` must be strictly positive.")
        if self.omega.shape != (self.n_nodes,):
            raise ValueError(f"`omega` must be of shape ({self.n_nodes},).")

    @staticmethod
    def available_backends() -> list[str]:
        available = ["python"]
        for backend, module_name in _BACKEND_MODULES.items():
            try:
                import_module(module_name)
            except ImportError:
                continue
            available.append(backend)
        return available

    def solve(self, theta0: ArrayLike, T: float, dt: float, backend: str = "python") -> tuple[FloatVector, FloatMatrix]:
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_order_parameter(backend)
        theta = kernel(self.omega, theta0_array, float(self.epsilon), float(dt), n_steps)
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta



# Classe 3 : model sparse (COO)

@dataclass(slots=True)
class SparseKuramotoModel(BaseModel):
    n_nodes: int
    n_edges: int
    edge_values: ArrayLike
    edge_rows: ArrayLike
    edge_cols: ArrayLike
    omega: ArrayLike
    epsilon: float
    name: str = field(default="kuramoto-sparse", init=False)
    row: NDArray = field(default=None, init=False)
    col: NDArray = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.edge_values = np.ascontiguousarray(self.edge_values, dtype=np.float64)
        self.edge_rows = np.ascontiguousarray(self.edge_rows, dtype=np.int32)
        self.edge_cols = np.ascontiguousarray(self.edge_cols, dtype=np.int32)
        self.omega = np.ascontiguousarray(self.omega, dtype=np.float64)
        self.row = self.edge_rows.copy()
        self.col = self.edge_cols.copy()
        
        if self.n_nodes <= 0:
            raise ValueError("`n_nodes` must be strictly positive.")
        if self.omega.shape != (self.n_nodes,):
            raise ValueError(f"`omega` must be of shape ({self.n_nodes},).")

    @staticmethod
    def available_backends() -> list[str]:
        available = ["python"]
        for backend, module_name in _BACKEND_MODULES.items():
            try:
                import_module(module_name)
            except ImportError:
                continue
            available.append(backend)
        return available

    def solve(self, theta0: ArrayLike, T: float, dt: float, backend: str = "python") -> tuple[FloatVector, FloatMatrix]:
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_sparse(backend)
        theta = kernel(
            self.edge_values, self.edge_rows, self.edge_cols,
            self.omega, theta0_array,
            float(self.epsilon), float(dt), n_steps
        )
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta