"""High-level Python interface for Kuramoto models."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..base import BaseModel

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


# BACKENDS DISPONIBLES

_BACKEND_MODULES = {
    "cython": "neuromass.models.kuramoto._native.cython_backend",
    "c": "neuromass.models.kuramoto._native.c_backend",
    "cpp": "neuromass.models.kuramoto._native.cpp_backend",
}

_BACKENDS_ = ["python", "cython", "c", "cpp"]
_ALL_BACKENDS = _BACKENDS_


# KERNELS PYTHON DE REFERENCE

def _python_kernel_naive(adjacency, omega, theta0, epsilon, dt, n_steps):
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


def _python_kernel_order_parameter(omega, theta0, epsilon, dt, n_steps):
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


def _python_kernel_sparse(edge_values, edge_rows, edge_cols, omega, theta0,
                          epsilon, dt, n_steps):
    n_nodes = omega.shape[0]
    theta = np.zeros((n_nodes, n_steps + 1), dtype=np.float64)
    theta[:, 0] = theta0

    is_csr = (edge_rows.shape[0] == n_nodes + 1)

    for step in range(n_steps):
        coupling = np.zeros(n_nodes)
        if is_csr:
            for i in range(n_nodes):
                theta_i = theta[i, step]
                c_i = 0.0
                for k in range(edge_rows[i], edge_rows[i + 1]):
                    j = edge_cols[k]
                    w = edge_values[k]
                    c_i += w * np.sin(theta[j, step] - theta_i)
                coupling[i] = c_i
        else:
            for e in range(edge_values.shape[0]):
                i = edge_rows[e]
                j = edge_cols[e]
                w = edge_values[e]
                coupling[i] += w * np.sin(theta[j, step] - theta[i, step])

        theta[:, step + 1] = theta[:, step] + dt * (
            omega + (epsilon / n_nodes) * coupling
        )
    return theta


# CHARGEURS DE BACKENDS  —  VERSIONS SÉQUENTIELLES

# CHARGEURS DE BACKENDS  —  VERSIONS OMP

def _load_backend_naive(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_naive
    if backend not in _ALL_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")

    module = import_module(_BACKEND_MODULES[backend])
    return module.simulate_naive_kuramoto_omp   #  OMP


def _load_backend_order_parameter(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_order_parameter
    if backend not in _ALL_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")

    module = import_module(_BACKEND_MODULES[backend])
    return module.simu_para_complexe_omp   #  OMP


_load_backend_global = _load_backend_order_parameter


def _load_backend_sparse(backend: str) -> Callable:
    if backend == "python":
        return _python_kernel_sparse
    if backend not in _ALL_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'.")

    module = import_module(_BACKEND_MODULES[backend])

    def _sparse_wrapper(edge_values, edge_rows, edge_cols, omega, theta0,
                        epsilon, dt, n_steps):
        row = edge_rows
        col = edge_cols
        n_edges = int(col.shape[0])

        if backend == "cython":
            return module.simu_sparse_omp(   #  OMP
                edge_values, row, col,
                omega, theta0,
                epsilon, dt, n_steps, n_edges,
                row, col,
            )
        return module.simu_sparse_omp(   #  OMP (C / C++)
            edge_values, row, col, omega, theta0,
            epsilon, dt, n_steps, n_edges,
            row, col,
        )

    return _sparse_wrapper


# CLASSE 1 : Naive Kuramoto

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
            raise ValueError(
                f"`adjacency` must be of shape ({self.n_nodes}, {self.n_nodes})."
            )

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

    def solve(self, theta0, T: float, dt: float, backend: str = "python"):
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_naive(backend)
        theta = kernel(
            self.adjacency, self.omega, theta0_array,
            float(self.epsilon), float(dt), n_steps
        )
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta


# CLASSE 2 : Mean-Field Kuramoto

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

    def solve(self, theta0, T: float, dt: float, backend: str = "python"):
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_order_parameter(backend)
        theta = kernel(
            self.omega, theta0_array,
            float(self.epsilon), float(dt), n_steps
        )
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta


# CLASSE 3 : Sparse Kuramoto (CSR)

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
    val: NDArray = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.edge_values = np.ascontiguousarray(self.edge_values, dtype=np.float64)
        self.edge_rows = np.ascontiguousarray(self.edge_rows, dtype=np.int32)
        self.edge_cols = np.ascontiguousarray(self.edge_cols, dtype=np.int32)
        self.omega = np.ascontiguousarray(self.omega, dtype=np.float64)

        if self.n_nodes <= 0:
            raise ValueError("`n_nodes` must be strictly positive.")
        if self.omega.shape != (self.n_nodes,):
            raise ValueError(f"`omega` must be of shape ({self.n_nodes},).")

        if self.edge_rows.shape[0] == self.n_nodes + 1:
            self.row = self.edge_rows.copy()
            self.col = self.edge_cols.copy()
            self.val = self.edge_values.copy()
        else:
            A = csr_matrix(
                (self.edge_values, (self.edge_rows, self.edge_cols)),
                shape=(self.n_nodes, self.n_nodes),
            )
            self.row = A.indptr.astype(np.int32)
            self.col = A.indices.astype(np.int32)
            self.val = A.data.astype(np.float64)

        self.n_edges = int(self.col.shape[0])

        assert self.row.shape[0] == self.n_nodes + 1
        assert self.row[-1] == self.n_edges

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

    def solve(self, theta0, T: float, dt: float, backend: str = "python"):
        theta0_array = np.ascontiguousarray(theta0, dtype=np.float64)
        if theta0_array.shape != (self.n_nodes,):
            raise ValueError(f"`theta0` must be of shape ({self.n_nodes},).")
        n_steps = int(round(T / dt))
        kernel = _load_backend_sparse(backend)
        theta = kernel(
            self.val, self.row, self.col,
            self.omega, theta0_array,
            float(self.epsilon), float(dt), n_steps
        )
        time = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
        return time, theta