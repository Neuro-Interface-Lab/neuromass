#include "kuramoto_c_kernel.h"
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Déclaration de la fonction CUDA
#ifdef __cplusplus
extern "C" {//Il permet de dire au compilateur C++ de ne pas modifier les noms des fonctions (name mangling), afin que le code C puisse les appeler.
#endif
void simulate_naive_kuramoto_cuda(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);
void simu_para_complexe_cuda(
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
);

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
);
#ifdef __cplusplus
}
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
) {
    int stride = n_steps + 1;
    printf("CPU simulate_naive_kuramoto_cpu appelée\n");
    printf("   N = %d, steps = %d\n", n_nodes, n_steps);

    // Copie initiale
    #pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }

    // Boucle temporelle avec OpenMP
    for (int step = 0; step < n_steps; ++step) {
        #pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i) {
            float coupling = 0.0f;
            float theta_i = output[i * stride + step];
            
            for (int j = 0; j < n_nodes; ++j) {
                float weight = adjacency[i * n_nodes + j];
                if (weight != 0.0f) {
                    coupling += weight * sinf(output[j * stride + step] - theta_i);
                }
            }
            
            output[i * stride + step + 1] = theta_i + dt * (
                omega[i] + (epsilon / n_nodes) * coupling
            );
        }
    }
    
    printf("CPU Version CPU terminée pour N = %d\n", n_nodes);
}




void simulate_naive_kuramoto_gpu(
    const float* adjacency,
    const float* omega,
    const float* theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float* output
) {
    printf("GPU simulate_naive_kuramoto_gpu appelée\n");
    printf("   N = %d, steps = %d\n", n_nodes, n_steps);
    
    // Vérifier si CUDA est disponible
    int cuda_available = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_available);
    
    if (err != cudaSuccess || cuda_available == 0) {
        printf("GPU CUDA non disponible ! Utilisation du CPU à la place.\n");
        simulate_naive_kuramoto_cpu(adjacency, omega, theta0, epsilon, dt, n_nodes, n_steps, output);
        return;
    }
    
    // Copie initiale (CPU)
    int stride = n_steps + 1;
    for (int i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }
    
    // Appeler le wrapper CUDA
    simulate_naive_kuramoto_cuda(
        adjacency, omega, theta0, epsilon, dt,
        n_nodes, n_steps, output
    );
    
    printf("GPU Version GPU terminée pour N = %d\n", n_nodes);
}

void simu_sparse_cpu(
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
    float *output)
{
    int i, j, e, step;
    int stride;
    float *coupling;

    stride = n_steps + 1;
    coupling = malloc(n_nodes * sizeof(float));

    if (coupling == NULL) {
        return;
    }

    #pragma omp parallel for
    for (i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step) {
        #pragma omp parallel for
        for (i = 0; i < n_nodes; ++i) {
            coupling[i] = 0.0f;
        }
        for (e = 0; e < n_edges; ++e) {
            i = edge_rows[e];
            j = edge_cols[e];
            float weight = edge_values[e];
            float theta_i = output[i * stride + step];
            float theta_j = output[j * stride + step];
            coupling[i] += weight * sinf(theta_j - theta_i);
        }
        #pragma omp parallel for
        for (i = 0; i < n_nodes; ++i) {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (
                omega[i] + (epsilon / n_nodes) * coupling[i]
            );
        }
    }

    free(coupling);
}

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
) {
    printf("GPU simu_sparse_gpu appelée\n");
    printf("   N = %d, steps = %d, edges = %d\n", n_nodes, n_steps, n_edges);
    
    // Vérifier si CUDA est disponible
    int cuda_available = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_available);
    
    if (err != cudaSuccess || cuda_available == 0) {
        printf("GPU CUDA non disponible ! Utilisation du CPU.\n");
        simu_sparse_cpu(edge_values, edge_rows, edge_cols, omega, theta0, 
                    epsilon, dt, n_nodes, n_steps, n_edges, row, col, output);
        return;
    }
    
    // Copie initiale (CPU)
    int stride = n_steps + 1;
    for (int i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }
    
    // Appeler le wrapper CUDA sparse
    simu_sparse_cuda(
        edge_values, edge_rows, edge_cols, omega, theta0, epsilon, dt,
        n_nodes, n_steps, n_edges, row, col, output
    );
    
    printf("GPU Version GPU sparse terminée pour N = %d\n", n_nodes);
}
void simu_para_complexe(
    const float *omega,
    const float *theta0,
    float epsilon,
    float dt,
    int n_nodes,
    int n_steps,
    float *output)
{
    int i, step;
    int stride = n_steps + 1;

    // Copie initiale (CPU) 
    #pragma omp parallel for
    for (i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }

    // Vérifier si on doit utiliser le GPU 
    int use_gpu = 0;
    if (n_nodes > 1000) {
        int cuda_available = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_available);
        if (err == cudaSuccess && cuda_available > 0) {
            use_gpu = 1;
            printf(" [MEAN-FIELD] Utilisation du GPU pour N = %d\n", n_nodes);
        } else {
            printf(" [MEAN-FIELD] CUDA non disponible, utilisation CPU\n");
        }
    }

    // --- Si GPU disponible et N grand → utiliser CUDA ---
    if (use_gpu) {
        simu_para_complexe_cuda(
            omega, theta0, epsilon, dt, n_nodes, n_steps, output
        );
        return;
    }

    // --- SINON : Version CPU (OpenMP) ---
    for (step = 0; step < n_steps; ++step) {
        float c = 0.0f;
        float s = 0.0f;
        float r, psi;

        #pragma omp parallel for reduction(+ : c, s)
        for (i = 0; i < n_nodes; ++i) {
            c += cosf(output[i * stride + step]);
            s += sinf(output[i * stride + step]);
        }
        c = c / n_nodes;
        s = s / n_nodes;
        r = sqrtf(c * c + s * s);
        psi = atan2f(s, c);

        #pragma omp parallel for
        for (int j = 0; j < n_nodes; ++j) {
            float theta_current = output[j * stride + step];
            output[j * stride + step + 1] = theta_current + dt * (
                omega[j] + epsilon * r * sinf(psi - theta_current)
            );
        }
    }
}