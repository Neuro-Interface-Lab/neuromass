#ifndef NEUROMASS_KURAMOTO_CUDA_KERNEL_H
#define NEUROMASS_KURAMOTO_CUDA_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void simulate_naive_kuramoto_cuda(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
);

#ifdef __cplusplus
}
#endif

#endif
