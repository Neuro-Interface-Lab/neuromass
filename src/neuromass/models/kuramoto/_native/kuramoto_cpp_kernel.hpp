#ifndef NEUROMASS_KURAMOTO_CPP_KERNEL_HPP
#define NEUROMASS_KURAMOTO_CPP_KERNEL_HPP


void simulate_naive_kuramoto_cpp(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output

);
void simu_para_complexe(
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
);

void simu_sparse(
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
    const int*row,
    const int*col,
    double* output
);

#endif
