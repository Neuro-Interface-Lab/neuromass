#include "kuramoto_cuda_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Déclaration du noyau dense
__global__ void kernel_naive_kuramoto_cuda(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
);


// WRAPPER DENSE AVEC extern "C"

extern "C" void simulate_naive_kuramoto_cuda(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
) {
    printf(" [CUDA WRAPPER] simulate_naive_kuramoto_cuda appelée\n");
    printf("   N=%d, steps=%d\n", n_nodes, n_steps);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("   Lancement : %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    
    kernel_naive_kuramoto_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        adjacency, omega, theta0, epsilon, dt, n_nodes, n_steps, output
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf(" [CUDA ERROR] %s\n", cudaGetErrorString(err));
    } else {
        printf("[CUDA WRAPPER] Noyau terminé avec succès\n");
    }
}
