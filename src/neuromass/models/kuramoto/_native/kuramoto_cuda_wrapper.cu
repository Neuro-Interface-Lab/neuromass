#include "kuramoto_cuda_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Déclaration du noyau dense
__global__ void kernel_naive_kuramoto_cuda(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);

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
);

__global__ void kernel_meanfield_sums(
    const float* theta,
    float* S,
    float* C,
    int n_nodes,
    int step,
    int stride
);

__global__ void kernel_meanfield_update(
    const float* omega,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output,
    float S,
    float C,
    int step
);



void simulate_naive_kuramoto_cuda(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
) {
    printf("[CUDA WRAPPER] simulate_naive_kuramoto_cuda appelée\n");
    printf("   N=%d, steps=%d\n", n_nodes, n_steps);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("   Lancement : %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    
    kernel_naive_kuramoto_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        adjacency, omega, theta0, epsilon, dt, n_nodes, n_steps, output
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
    } else {
        printf("[CUDA WRAPPER] Noyau terminé avec succès\n");
    }
}

void simu_sparse_cuda(
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
    printf(" simu_sparse_cuda appelée\n");
    printf("   N=%d, steps=%d, edges=%d\n", n_nodes, n_steps, n_edges);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("   Lancement : %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    
    kernel_sparse_kuramoto_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        edge_values, edge_rows, edge_cols, omega, theta0, epsilon, dt,
        n_nodes, n_steps, n_edges, row, col, output
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s\n", cudaGetErrorString(err));
    } else {
        printf("[CUDA WRAPPER] Noyau sparse terminé avec succès\n");
    }
}
// ============================================================
// WRAPPER MEAN-FIELD AVEC extern "C"
// ============================================================
extern "C" void simu_para_complexe_cuda(
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
) {
    printf("  simu_para_complexe_cuda appelée\n");
    printf("   N=%d, steps=%d\n", n_nodes, n_steps);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    int stride = n_steps + 1;
    
    // Allouer S et C sur GPU
    float *d_S, *d_C;
    cudaMalloc(&d_S, sizeof(float));
    cudaMalloc(&d_C, sizeof(float));
    
    // Copie initiale des phases
    for (int i = 0; i < n_nodes; i++) {
        output[i * stride] = theta0[i];
    }
    
    // Boucle sur les pas de temps
    for (int step = 0; step < n_steps; step++) {
        // Réinitialiser S et C à 0
        cudaMemset(d_S, 0, sizeof(float));
        cudaMemset(d_C, 0, sizeof(float));
        
        // Calculer S et C sur GPU
        kernel_meanfield_sums<<<blocksPerGrid, threadsPerBlock>>>(
            output, d_S, d_C, n_nodes, step, stride
        );
        
        // Récupérer S et C sur CPU
        float S, C;
        cudaMemcpy(&S, d_S, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&C, d_C, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Mettre à jour les phases
        kernel_meanfield_update<<<blocksPerGrid, threadsPerBlock>>>(
            omega, epsilon, dt, n_nodes, n_steps, output, S, C, step
        );
    }
    
    cudaFree(d_S);
    cudaFree(d_C);
    
    printf("[CUDA WRAPPER] Mean-field terminé avec succès\n");
}