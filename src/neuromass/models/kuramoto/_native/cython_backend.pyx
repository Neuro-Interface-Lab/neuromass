"""Cython implementation of the naive Kuramoto solver."""

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sin

cnp.import_array()


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
    cdef cnp.ndarray[cnp.float64_t, ndim=2] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    for i in range(n_nodes):
        theta[i, 0] = theta0[i]

    for step in range(n_steps):
        for i in range(n_nodes):
            coupling = 0.0
            for j in range(n_nodes):
                if adjacency[i, j] != 0.0:
                    coupling += adjacency[i, j] * sin(theta[j, step] - theta[i, step])
            theta[i, step + 1] = theta[i, step] + dt * (
                omega[i] + (epsilon / n_nodes) * coupling
            )

    return theta

