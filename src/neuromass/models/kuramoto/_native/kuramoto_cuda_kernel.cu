#include <cuda_runtime.h>
#include <math.h>

// NOYAU DENSE
__global__ void kernel_naive_kuramoto_cuda(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int stride = n_steps + 1;
    output[i * stride] = theta0[i];
    
    for (int step = 0; step < n_steps; step++) {
        float coupling = 0.0;
        float theta_i = output[i * stride + step];
        
        for (int j = 0; j < n_nodes; j++) {
            float weight = adjacency[i * n_nodes + j];
            if (weight != 0.0) {
                coupling += weight * sinf(output[j * stride + step] - theta_i);
            }
        }
        
        output[i * stride + step + 1] = theta_i + dt * (omega[i] + (epsilon / n_nodes) * coupling);
    }
}

// NOYAU SPARSE CSR
__global__ void kernel_sparse_kuramoto_cuda(
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
    const int* row,
    const int* col,
    float* output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int stride = n_steps + 1;
    output[i * stride] = theta0[i];
    
    for (int step = 0; step < n_steps; step++) {
        float coupling = 0.0;
        float theta_i = output[i * stride + step];
        
        for (int idx = row[i]; idx < row[i+1]; idx++) {
            int j = col[idx];
            float weight = edge_values[idx];
            coupling += weight * sinf(output[j * stride + step] - theta_i);
        }
        
        output[i * stride + step + 1] = theta_i + dt * (omega[i] + (epsilon / n_nodes) * coupling);
    }
}
// NOYAU parametre complexe



// NOYAU MEAN-FIELD - S ET C SUR GPU 
__global__ void kernel_meanfield_sums(
    const float* theta,
    float* S,
    float* C,
    int n_nodes,
    int step,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    float theta_i = theta[i * stride + step];

    // Accumulation atomique (simple mais efficace pour ce cas)
    atomicAdd(S, sinf(theta_i));
    atomicAdd(C, cosf(theta_i));
}
// Noyau pour mettre à jour les phases avec S et C
__global__ void kernel_meanfield_update(
    const float* omega,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output,
    float S,                 // ← Somme des sin (calculée sur GPU)
    float C,                 // ← Somme des cos (calculée sur GPU)
    int step
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int stride = n_steps + 1;
    float theta_i = output[i * stride + step];
    
    // Paramètre d'ordre
    float r = sqrtf(S*S + C*C) / n_nodes;
    float psi = atan2f(S, C);
    
    output[i * stride + step + 1] = theta_i + dt * (omega[i] + epsilon * r * sinf(psi - theta_i));
}