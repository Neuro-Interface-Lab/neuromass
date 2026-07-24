#include <cuda_runtime.h>
#include <math.h>

// NOYAU DENSE
__global__ void kernel_naive_kuramoto_cuda(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int stride = n_steps + 1;
    output[i * stride] = theta0[i];
    
    for (int step = 0; step < n_steps; step++) {
        double coupling = 0.0;
        double theta_i = output[i * stride + step];
        
        for (int j = 0; j < n_nodes; j++) {
            double weight = adjacency[i * n_nodes + j];
            if (weight != 0.0) {
                coupling += weight * sin(output[j * stride + step] - theta_i);
            }
        }
        
        output[i * stride + step + 1] = theta_i + dt * (omega[i] + (epsilon / n_nodes) * coupling);
    }
}

// NOYAU SPARSE CSR
__global__ void kernel_sparse_kuramoto_cuda(
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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int stride = n_steps + 1;
    output[i * stride] = theta0[i];
    
    for (int step = 0; step < n_steps; step++) {
        double coupling = 0.0;
        double theta_i = output[i * stride + step];
        
        for (int idx = row[i]; idx < row[i+1]; idx++) {
            int j = col[idx];
            double weight = edge_values[idx];
            coupling += weight * sin(output[j * stride + step] - theta_i);
        }
        
        output[i * stride + step + 1] = theta_i + dt * (omega[i] + (epsilon / n_nodes) * coupling);
    }
}
