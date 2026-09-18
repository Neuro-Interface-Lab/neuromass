"""Cython implementation of the Kuramoto solver (sequential + OpenMP)."""

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, atan2, sqrt
from cython.parallel cimport prange

cnp.import_array()


# ============================================================
# VERSION 1 : NAIVE (DENSE) - SEQUENTIEL
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_naive_kuramoto(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] adjacency,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version séquentielle du solveur naïf (dense)."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, j, step
    cdef double coupling_i
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[:, ::1] adj = adjacency
    cdef double[::1] omega_v = omega

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        for i in range(n_nodes):
            coupling_i = 0.0
            for j in range(n_nodes):
                if adj[i, j] != 0.0:
                    coupling_i = coupling_i + adj[i, j] * sin(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling_i
            )

    return theta_arr


# ============================================================
# VERSION 2 : NAIVE (DENSE) - OPENMP
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_naive_kuramoto_omp(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] adjacency,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version OpenMP du solveur naïf (dense)."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, j, step
    cdef double coupling_i
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[:, ::1] adj = adjacency
    cdef double[::1] omega_v = omega

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        for i in prange(n_nodes, nogil=True):
            coupling_i = 0.0
            for j in range(n_nodes):
                if adj[i, j] != 0.0:
                    coupling_i = coupling_i + adj[i, j] * sin(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling_i
            )

    return theta_arr


# ============================================================
# VERSION 3 : MEAN-FIELD - SEQUENTIEL
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_para_complexe(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version séquentielle du solveur mean-field."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, step
    cdef double c, s, r, psi
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[::1] omega_v = omega

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        c = 0.0
        s = 0.0

        for i in range(n_nodes):
            c += cos(theta[i, step])
            s += sin(theta[i, step])
        c = c / n_nodes
        s = s / n_nodes
        r = sqrt(c * c + s * s)
        psi = atan2(s, c)

        for i in range(n_nodes):
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + epsilon * r * sin(psi - theta[i, step])
            )

    return theta_arr


# ============================================================
# VERSION 4 : MEAN-FIELD - OPENMP
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_para_complexe_omp(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version OpenMP du solveur mean-field."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, step
    cdef double c, s, r, psi
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[::1] omega_v = omega

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        c = 0.0
        s = 0.0

        for i in range(n_nodes):
            c += cos(theta[i, step])
            s += sin(theta[i, step])
        c = c / n_nodes
        s = s / n_nodes
        r = sqrt(c * c + s * s)
        psi = atan2(s, c)

        for i in prange(n_nodes, nogil=True):
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + epsilon * r * sin(psi - theta[i, step])
            )

    return theta_arr


# ============================================================
# VERSION 5 : SPARSE CSR - SEQUENTIEL
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_sparse(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] edge_values,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_rows,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_cols,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
    int n_edges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] row,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] col,
):
    """Version séquentielle du solveur sparse (CSR)."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, k, j, step
    cdef double coupling_i, weight, theta_i
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[::1] omega_v = omega
    cdef double[::1] val = edge_values
    cdef int[::1] row_v = row
    cdef int[::1] col_v = col

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        # ✅ CSR : parcourt les voisins de chaque nœud
        for i in range(n_nodes):
            theta_i = theta[i, step]
            coupling_i = 0.0
            for k in range(row_v[i], row_v[i + 1]):
                j = col_v[k]
                weight = val[k]
                coupling_i = coupling_i + weight * sin(theta[j, step] - theta_i)
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling_i
            )

    return theta_arr


# ============================================================
# VERSION 6 : SPARSE CSR - OPENMP
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_sparse_omp(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] edge_values,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_rows,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_cols,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
    int n_edges,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] row,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] col,
):
    """Version OpenMP du solveur sparse (CSR)."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, k, j, step
    cdef double coupling_i, weight, theta_i
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[::1] omega_v = omega
    cdef double[::1] val = edge_values
    cdef int[::1] row_v = row
    cdef int[::1] col_v = col

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        # ✅ CSR parallélisé
        for i in prange(n_nodes, nogil=True):
            theta_i = theta[i, step]
            coupling_i = 0.0
            for k in range(row_v[i], row_v[i + 1]):
                j = col_v[k]
                weight = val[k]
                coupling_i = coupling_i + weight * sin(theta[j, step] - theta_i)
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling_i
            )

    return theta_arr