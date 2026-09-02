"""Cython implementation of the naive Kuramoto solver."""

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sinf, cosf, atan2f, sqrtf  
from cython.parallel cimport prange

cnp.import_array()


# VERSION NAÏVE (DENSE)

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_naive_kuramoto(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] adjacency,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] theta0,
    float epsilon,
    float dt,
    int n_steps,
):
    """Explicit Euler solver for the non-delayed Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, j, step
    cdef float coupling
    cdef cnp.ndarray[cnp.float32_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float32, 
    )
    cdef float[:, ::1] theta = theta_arr
    cdef float[:, ::1] adj = adjacency
    cdef float[::1] omega_v = omega

    # Condition initiale
    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    # Boucle temporelle
    for step in range(n_steps):
        for i in prange(n_nodes, nogil=True):
            coupling = 0.0
            for j in range(n_nodes):
                if adj[i, j] != 0.0:
                    coupling = coupling + adj[i, j] * sinf(theta[j, step] - theta[i, step]) 
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling
            )

    return theta_arr


# VERSION MEAN-FIELD

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_para_complexe(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] theta0,
    float epsilon,
    float dt,
    int n_steps,
):
    """Explicit Euler solver for the mean-field Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, step
    cdef float c, s, r, psi
    cdef cnp.ndarray[cnp.float32_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float32,
    )
    cdef float[:, ::1] theta = theta_arr
    cdef float[::1] omega_v = omega

    # Condition initiale
    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    # Boucle temporelle
    for step in range(n_steps):
        c = 0.0
        s = 0.0

        for i in range(n_nodes):
            c += cosf(theta[i, step])
            s += sinf(theta[i, step])
        
        c /= n_nodes
        s /= n_nodes
        r = sqrtf(c * c + s * s)
        psi = atan2f(s, c)

        for i in prange(n_nodes, nogil=True):
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + epsilon * r * sinf(psi - theta[i, step])
            )

    return theta_arr

# VERSION SPARSE 

@cython.boundscheck(False)
@cython.wraparound(False)
def simu_sparse(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] theta0,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] edge_values,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_rows,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_cols,
    float epsilon,
    float dt,
    int n_steps,
):
    """Explicit Euler solver for the sparse Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int n_edges = edge_values.shape[0]
    cdef int i,j, e, step
    cdef float coupling_i, weight
    cdef cnp.ndarray[cnp.float32_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float32,
    )
    cdef float[:, ::1] theta = theta_arr
    cdef float[::1] omega_v = omega
    cdef float[::1] edge_values_v = edge_values
    cdef int[::1] edge_rows_v = edge_rows
    cdef int[::1] edge_cols_v = edge_cols

    # Condition initiale
    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    # Boucle temporelle
    for step in range(n_steps):
        for i in prange(n_nodes, nogil=True):
            coupling_i = 0.0
            for e in range(n_edges):
                if edge_rows_v[e] == i:
                    j = edge_cols_v[e]
                    weight = edge_values_v[e]
                    coupling_i =coupling_i+ weight * sinf(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling_i
            )

    return theta_arr