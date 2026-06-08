#ifndef NEUROMASS_KURAMOTO_C_KERNEL_H
#define NEUROMASS_KURAMOTO_C_KERNEL_H

void simulate_naive_kuramoto_c(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
);

#endif
