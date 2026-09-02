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

    float *d_adjacency, *d_omega, *d_theta0, *d_output;
    
    cudaMalloc(&d_adjacency, n_nodes * n_nodes * sizeof(float));
    cudaMalloc(&d_omega, n_nodes * sizeof(float));
    cudaMalloc(&d_theta0, n_nodes * sizeof(float));
    cudaMalloc(&d_output, n_nodes * (n_steps + 1) * sizeof(float));

    // Copies CPU → GPU
    cudaMemcpy(d_adjacency, adjacency, n_nodes * n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, omega, n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta0, theta0, n_nodes * sizeof(float), cudaMemcpyHostToDevice);

    printf("[CUDA WRAPPER] simulate_naive_kuramoto_cuda appelée\n");
    printf("   N=%d, steps=%d\n", n_nodes, n_steps);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("   Lancement : %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    
    kernel_naive_kuramoto_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        d_adjacency, d_omega, d_theta0, epsilon, dt, n_nodes, n_steps, d_output
    );
    
    cudaDeviceSynchronize();
    
    // Copie GPU → CPU
    cudaMemcpy(output, d_output, n_nodes * (n_steps + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Libération
    cudaFree(d_adjacency);
    cudaFree(d_omega);
    cudaFree(d_theta0);
    cudaFree(d_output);
    printf("CUDA WRAPPER : Noyau terminé avec succès\n");
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

    float *d_edge_values, *d_omega, *d_theta0, *d_output;
    int *d_edge_rows, *d_edge_cols;
    
    cudaMalloc(&d_edge_values, n_edges * sizeof(float));
    cudaMalloc(&d_edge_rows, n_edges * sizeof(int));
    cudaMalloc(&d_edge_cols, n_edges * sizeof(int));
    cudaMalloc(&d_omega, n_nodes * sizeof(float));
    cudaMalloc(&d_theta0, n_nodes * sizeof(float));
    cudaMalloc(&d_output, n_nodes * (n_steps + 1) * sizeof(float));

    printf(" simu_sparse_cuda appelée\n");
    printf("   N=%d, steps=%d, edges=%d\n", n_nodes, n_steps, n_edges);

    // Copies CPU → GPU
    cudaMemcpy(d_edge_values, edge_values, n_edges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_rows, edge_rows, n_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_cols, edge_cols, n_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, omega, n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta0, theta0, n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("   Lancement : %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    int *d_row, *d_col;
    cudaMalloc(&d_row, (n_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col, n_edges * sizeof(int));
    cudaMemcpy(d_row, row, (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, n_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    kernel_sparse_kuramoto_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        d_edge_values, d_edge_rows, d_edge_cols, d_omega, d_theta0, epsilon, dt,n_nodes, n_steps, n_edges, d_row, d_col, d_output

    );
    
    cudaDeviceSynchronize();
    
    // Copie GPU → CPU
    cudaMemcpy(output, d_output, n_nodes * (n_steps + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Libération
    cudaFree(d_edge_values);
    cudaFree(d_edge_rows);
    cudaFree(d_edge_cols);
    cudaFree(d_omega);
    cudaFree(d_theta0);
    cudaFree(d_output);
}

// WRAPPER MEAN-FIELD AVEC extern "C"

extern "C" void simu_para_complexe_cuda(
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
) {

    float *d_omega, *d_theta0, *d_output, *d_S, *d_C;
    int stride = n_steps + 1;
    
    cudaMalloc(&d_omega, n_nodes * sizeof(float));
    cudaMalloc(&d_theta0, n_nodes * sizeof(float));
    cudaMalloc(&d_output, n_nodes * stride * sizeof(float));  
    cudaMalloc(&d_S, sizeof(float));
    cudaMalloc(&d_C, sizeof(float));

    printf("  simu_para_complexe_cuda appelée\n");
    printf("   N=%d, steps=%d\n", n_nodes, n_steps);

    cudaMemcpy(d_omega, omega, n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta0, theta0, n_nodes * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < n_nodes; i++) {
        cudaMemcpy(d_output + i * stride, d_theta0 + i, sizeof(float), cudaMemcpyDeviceToDevice);
    }


    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    
    for (int step = 0; step < n_steps; step++) {
        cudaMemset(d_S, 0, sizeof(float));
        cudaMemset(d_C, 0, sizeof(float));
        
        kernel_meanfield_sums<<<blocksPerGrid, threadsPerBlock>>>(
            d_output, d_S, d_C, n_nodes, step, stride );
        
        float S, C;
        cudaMemcpy(&S, d_S, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&C, d_C, sizeof(float), cudaMemcpyDeviceToHost);
        
        kernel_meanfield_update<<<blocksPerGrid, threadsPerBlock>>>(
            d_omega, epsilon, dt, n_nodes, n_steps, d_output, S, C, step
        );
    }
    cudaDeviceSynchronize();
    
    // Copie GPU → CPU
    cudaMemcpy(output, d_output, n_nodes * (n_steps + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Libération
    cudaFree(d_omega);
    cudaFree(d_theta0);
    cudaFree(d_output);
    cudaFree(d_S);
    cudaFree(d_C);
    
    printf("[CUDA WRAPPER] Mean-field terminé avec succès\n");
}