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

#endif
