#include "kuramoto_c_kernel.h"
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Déclaration de la fonction CUDA
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




void simulate_naive_kuramoto_c(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
) {
    int i, j, step;
    int stride = n_steps + 1;
    double coupling, weight;
    printf(" simulate_naive_kuramoto_c appelée avec N = %d\n", n_nodes);


    // --- Copie initiale (CPU) ---
    #pragma omp parallel for
    for (i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }
    simulate_naive_kuramoto_cuda(
        adjacency, omega, theta0, epsilon, dt,
        n_nodes, n_steps, output
    );}


   
void simu_para_complexe(
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double *output)
{
    int i;
    int j;
    int step;
    int stride = n_steps + 1;

#pragma omp parallel for
    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step)
    {
        double c = 0.0;
        double s = 0.0;
        double r;
        double psi;
        double theta_current;
#pragma omp parallel for reduction(+ : c, s)
        for (i = 0; i < n_nodes; ++i)
        {
            c += cos(output[i * stride + step]);
            s += sin(output[i * stride + step]);
        }
        c = c / n_nodes;
        s = s / n_nodes;
        r = sqrt(c * c + s * s);

        psi = atan2(s, c);
#pragma omp parallel for private(theta_current)
        for (j = 0; j < n_nodes; ++j)
        {
            double theta_current = output[j * stride + step];
            output[j * stride + step + 1] = theta_current + dt * (omega[j] + epsilon * r * sin(psi - theta_current));
        }
    }
}

void simu_sparse(
    const double *edge_values,
    const int *edge_rows,
    const int *edge_cols,
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    int n_edges,
    const int *row,
    const int *col,
    double *output)
{
    int i;
    int j;
    int e;
    int step;
    int stride;
    double *coupling;

    stride = n_steps + 1;
    coupling = malloc(n_nodes * sizeof(double));

    if (coupling == NULL)
    {
        return;
    }
#pragma omp parallel for
    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step)
    {
#pragma omp parallel for
        for (i = 0; i < n_nodes; ++i)
        {
            coupling[i] = 0.0;
        }
        for (e = 0; e < n_edges; ++e)
        {
            i = edge_rows[e];
            j = edge_cols[e];
            double weight = edge_values[e];
            double theta_i = output[i * stride + step];
            double theta_j = output[j * stride + step];
            coupling[i] += weight * sin(theta_j - theta_i);
        }
#pragma omp parallel for
        for (i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
    }

    free(coupling);
}
