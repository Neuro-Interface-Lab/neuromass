#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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

int main() {
    int N = 10;
    int steps = 5;
    double epsilon = 1.0;
    double dt = 0.01;
    
    printf("TEST DU KERNEL KURAMOTO CUDA\n");

    printf("N = %d, steps = %d\n\n", N, steps);
    
    // --- Allocation CPU ---
    double *h_adj = (double*)malloc(N * N * sizeof(double));
    double *h_omega = (double*)malloc(N * sizeof(double));
    double *h_theta0 = (double*)malloc(N * sizeof(double));
    double *h_output = (double*)malloc(N * (steps + 1) * sizeof(double));
    
    // --- Initialisation ---
    for (int i = 0; i < N; i++) {
        h_omega[i] = 1.0;
        h_theta0[i] = (double)i / N * 2 * 3.14159;
        for (int j = 0; j < N; j++) {
            h_adj[i * N + j] = (i == j) ? 0.0 : 1.0;
        }
    }
    
    // --- Allocation GPU ---
    double *d_adj, *d_omega, *d_theta0, *d_output;
    cudaMalloc(&d_adj, N * N * sizeof(double));
    cudaMalloc(&d_omega, N * sizeof(double));
    cudaMalloc(&d_theta0, N * sizeof(double));
    cudaMalloc(&d_output, N * (steps + 1) * sizeof(double));
    
    // --- Copie CPU → GPU ---
    cudaMemcpy(d_adj, h_adj, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, h_omega, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta0, h_theta0, N * sizeof(double), cudaMemcpyHostToDevice);
    
    // --- Lancement du noyau ---
    printf("Lancement du kernel sur GPU...\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_naive_kuramoto_cuda(d_adj, d_omega, d_theta0, epsilon, dt, N, steps, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float temps_ms;
    cudaEventElapsedTime(&temps_ms, start, stop);
    
    // --- Copie GPU → CPU ---
    cudaMemcpy(h_output, d_output, N * (steps + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    
    // --- Affichage ---
    printf("Temps d'exécution : %.3f ms\n", temps_ms);
    printf("\nRésultats (theta après %d pas) :\n", steps);
    for (int i = 0; i < 3 && i < N; i++) {
        printf("  theta[%d] = %.6f\n", i, h_output[i * (steps + 1) + steps]);
    }
    
    // --- Vérification ---
    int ok = 1;
    for (int i = 0; i < N * (steps + 1); i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) {
            ok = 0;
            break;
        }
    }
    printf("\n%s\n", ok ? " Test réussi !" : "Erreur détectée");
    
    // --- Libération ---
    cudaFree(d_adj);
    cudaFree(d_omega);
    cudaFree(d_theta0);
    cudaFree(d_output);
    free(h_adj);
    free(h_omega);
    free(h_theta0);
    free(h_output);
    
    return 0;
}
