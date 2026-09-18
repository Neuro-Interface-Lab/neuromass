#ifndef NEUROMASS_KURAMOTO_C_KERNEL_H
#define NEUROMASS_KURAMOTO_C_KERNEL_H

#ifdef __cplusplus
extern "C"
{
#endif

    void simulate_naive_kuramoto_c(
        const double *adjacency,
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double *output);
    void simu_para_complexe_c(
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double *output);

    void simu_sparse_c(
        const double *edge_values,
        const int *edge_rows,
        const int *edge_cols,
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        int n_edges,
        const int *row,
        const int *col,
        double *output);

    void simulate_naive_kuramoto_c_omp(

        const double *adjacency,
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double *output);
    void simu_para_complexe_c_omp(
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        double *output);
    void simu_sparse_c_omp(
        const double *edge_values,
        const int *edge_rows,
        const int *edge_cols,
        const double *omega,
        const double *theta0,
        double epsilon,
        double dt,
        int n_nodes,
        int n_steps,
        int n_edges,
        const int *row,
        const int *col,
        double *output);

#ifdef __cplusplus
}
#endif

#endif
