"""Cython wrapper around the C++ implementation of the Kuramoto solver."""

import numpy as np
cimport numpy as cnp

cnp.import_array()

# =============================================
# DECLARATION DES FONCTIONS C++
# =============================================
cdef extern from "kuramoto_cpp_kernel.hpp":

    # --- Version séquentielle ---
    void simulate_naive_kuramoto_cpp(
        const double* adjacency,
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double* output
    )

    void simu_para_complexe_cpp(
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double* output
    )

    void simu_sparse_cpp(
        const double* edge_values,
        const int* edge_rows,
        const int* edge_cols,
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        int n_edges,
        const int* row,
        const int* col,
        double* output
    )

    # --- Version OpenMP ---
    void simulate_naive_kuramoto_cpp_omp(
        const double* adjacency,
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double* output
    )

    void simu_para_complexe_cpp_omp(
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double* output
    )

    void simu_sparse_cpp_omp(
        const double* edge_values,
        const int* edge_rows,
        const int* edge_cols,
        const double* omega,
        const double* theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        int n_edges,
        const int* row,
        const int* col,
        double* output
    )


# =============================================
# VERSION SEQUENTIELLE
# =============================================

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


def simu_para_complexe(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version séquentielle du solveur mean-field."""

    cdef int n_nodes = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simu_para_complexe_cpp(              # ← CORRIGÉ
        &omega[0],
        &theta0[0],
        epsilon,
        dt,
        n_nodes,
        n_steps,
        &theta[0, 0],
    )

    return theta


def simu_sparse(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] edge_values,
    int[::1] edge_rows,
    int[::1] edge_cols,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
    int n_edges,
    int[::1] row,
    int[::1] col,
):
    """Version séquentielle du solveur sparse (CSR)."""

    cdef int n_nodes = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simu_sparse_cpp(                     # ← CORRIGÉ
        &edge_values[0],
        <const int*> &edge_rows[0],
        <const int*> &edge_cols[0],
        &omega[0],
        &theta0[0],
        epsilon,
        dt,
        n_nodes,
        n_steps,
        n_edges,
        <const int*> &row[0],
        <const int*> &col[0],
        &theta[0, 0],
    )

    return theta


# =============================================
# VERSION OPENMP
# =============================================

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
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simulate_naive_kuramoto_cpp_omp(
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


def simu_para_complexe_omp(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
):
    """Version OpenMP du solveur mean-field."""

    cdef int n_nodes = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simu_para_complexe_cpp_omp(          # ← CORRIGÉ
        &omega[0],
        &theta0[0],
        epsilon,
        dt,
        n_nodes,
        n_steps,
        &theta[0, 0],
    )

    return theta


def simu_sparse_omp(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] edge_values,
    int[::1] edge_rows,
    int[::1] edge_cols,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta0,
    double epsilon,
    double dt,
    int n_steps,
    int n_edges,
    int[::1] row,
    int[::1] col,
):
    """Version OpenMP du solveur sparse (CSR)."""

    cdef int n_nodes = omega.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] theta = np.zeros(
        (n_nodes, n_steps + 1),
        dtype=np.float64,
    )

    simu_sparse_cpp_omp(                 # ← CORRIGÉ
        &edge_values[0],
        <const int*> &edge_rows[0],
        <const int*> &edge_cols[0],
        &omega[0],
        &theta0[0],
        epsilon,
        dt,
        n_nodes,
        n_steps,
        n_edges,
        <const int*> &row[0],
        <const int*> &col[0],
        &theta[0, 0],
    )

    return theta