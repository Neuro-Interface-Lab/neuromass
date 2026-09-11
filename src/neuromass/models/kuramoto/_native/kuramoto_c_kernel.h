#ifndef NEUROMASS_KURAMOTO_C_KERNEL_H
#define NEUROMASS_KURAMOTO_C_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif
void simulate_naive_kuramoto_cpu(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);
void simulate_naive_kuramoto_gpu(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);


void simu_para_complexe(
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);

void simu_sparse_cpu(
    const float* edge_values,
    const int* edge_rows,
    const int* edge_cols,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    int n_edges,
    const int*row,
    const int*col,
    float* output
);
void simu_sparse_gpu(
    const float *edge_values,
    const int *edge_rows,
    const int *edge_cols,
    const float *omega,
    const float *theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    int n_edges,
    const int *row,
    const int *col,
    float *output
);
#ifdef __cplusplus
}
#endif

#endif
