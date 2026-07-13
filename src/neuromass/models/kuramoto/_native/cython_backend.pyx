"""Cython implementation of the naive Kuramoto solver."""

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, atan2, sqrt

cnp.import_array()
from cython.parallel cimport prange

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
    """Explicit Euler solver for the non-delayed Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, j, step
    cdef double coupling
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
        for i in prange(n_nodes, nogil= True):
            coupling = 0.0
            for j in range(n_nodes):
                if adj[i, j] != 0.0:
                    coupling = coupling +adj[i, j] * sin(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega_v[i] + (epsilon / n_nodes) * coupling
            )

    return theta_arr




@cython.boundscheck(False)
@cython.wraparound(False)
def simu_para_complexe(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Explicit Euler solver for the non-delayed Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int i, j, step
    cdef double coupling, c, s, r, psi
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

        for i in prange(n_nodes, nogil= True):
            theta[i,step+1]= theta[i,step] + dt *(omega_v[i] + epsilon * r*sin(psi - theta[i,step]))
            

    return theta_arr




@cython.boundscheck(False)
@cython.wraparound(False)
def simu_sparse(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] edge_values,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_rows,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] edge_cols,
    double epsilon,
    double dt,
    int n_steps,
):
    """Explicit Euler solver for the non-delayed Kuramoto model."""

    cdef int n_nodes = omega.shape[0]
    cdef int n_edges = edge_values.shape[0]
    cdef int i, j, step
    cdef double coupling_i
    cdef int stride = n_steps + 1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta_arr = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )
    cdef double[:, ::1] theta = theta_arr
    cdef double[::1] omega_v = omega
    cdef double[::1] edge_values_v = edge_values
    cdef int[::1] edge_rows_v = edge_rows
    cdef int[::1] edge_cols_v = edge_cols

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]
    for step in prange(n_steps, nogil = True):

        for i in range(n_nodes):
            coupling_i = 0.0

            for e in range(n_edges):
                if edge_rows_v[e] == i:
                    j = edge_cols_v[e]
                    weight = edge_values_v[e]
                    coupling_i = coupling_i +weight * sin(theta[j,step]- theta[i, step])  

      
        theta[i,step+1]= theta[i,step] + dt *(omega_v[i] + (epsilon/n_nodes) * coupling_i)
            

    return theta_arr



