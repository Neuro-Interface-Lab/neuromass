"""Cython wrapper around the C++ implementation of the naive Kuramoto solver."""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "kuramoto_cpp_kernel.hpp":
    void simulate_naive_kuramoto_cpp(
        const double* adjacency,
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double* output,
    )


def simulate_naive_kuramoto(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] adjacency,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Explicit Euler solver backed by a C++ kernel."""

    cdef int n_nodes = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simulate_naive_kuramoto_cpp(
        &adjacency[0, 0],
        &omega[0],
        &theta0[0],
        epsilon,
        dt,
        n_nodes,
        n_steps,
        &theta[0, 0],
    )

    return theta

