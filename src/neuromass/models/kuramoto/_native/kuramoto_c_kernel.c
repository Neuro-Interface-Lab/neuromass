#include "kuramoto_c_kernel.h"
#include <omp.h>
#include <math.h>
#include <stdlib.h>

void simulate_naive_kuramoto_c(
    const double *adjacency,
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
    double coupling;
    double weight;

    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step)
    {
        for (i = 0; i < n_nodes; ++i)
        {
            coupling = 0.0;
            for (j = 0; j < n_nodes; ++j)
            {
                weight = adjacency[i * n_nodes + j];
                if (weight != 0.0)
                {
                    coupling += weight * sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling);
        }
    }
}
void simu_para_complexe_c(
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

        for (i = 0; i < n_nodes; ++i)
        {
            c += cos(output[i * stride + step]);
            s += sin(output[i * stride + step]);
        }
        c = c / n_nodes;
        s = s / n_nodes;
        r = sqrt(c * c + s * s);

        psi = atan2(s, c);
        for (j = 0; j < n_nodes; ++j)
        {
            double theta_current = output[j * stride + step];
            output[j * stride + step + 1] = theta_current + dt * (omega[j] + epsilon * r * sin(psi - theta_current));
        }
    }
}

void simu_sparse_c(
    const double *edge_values, // CSR : valeurs des arêtes
    const int *edge_rows,      // (non utilisé en CSR)
    const int *edge_cols,      // (non utilisé en CSR)
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    int n_edges,
    const int *row, // CSR : row[i] = début des voisins de i
    const int *col, // CSR : col[k] = indice du voisin
    double *output)
{
    int i, j, k, step;
    int stride = n_steps + 1;
    double *coupling = malloc(n_nodes * sizeof(double));

    if (coupling == NULL)
        return;

    // Condition initiale
    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    // Boucle temporelle
    for (step = 0; step < n_steps; ++step)
    {
        //  CSR : parcourt les voisins de chaque nœud
        for (i = 0; i < n_nodes; ++i)
        {
            double theta_i = output[i * stride + step];
            double c_i = 0.0;

            // Parcourt les voisins de i (entre row[i] et row[i+1])
            for (k = row[i]; k < row[i + 1]; ++k)
            {
                j = col[k];
                double weight = edge_values[k];
                c_i += weight * sin(output[j * stride + step] - theta_i);
            }

            coupling[i] = c_i;
        }

        // Mise à jour
        for (i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
    }

    free(coupling);
}

void simulate_naive_kuramoto_c_omp(

    const double *adjacency,
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
    double coupling;
    double weight;

#pragma omp parallel for
    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step)
    {
#pragma omp parallel for private(j, coupling, weight)
        for (i = 0; i < n_nodes; ++i)
        {
            coupling = 0.0;
            for (j = 0; j < n_nodes; ++j)
            {
                weight = adjacency[i * n_nodes + j];
                if (weight != 0.0)
                {
                    coupling += weight * sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling);
        }
    }
}
void simu_para_complexe_c_omp(
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

void simu_sparse_c_omp(
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
    int i, j, k, step;
    int stride = n_steps + 1;
    double *coupling = malloc(n_nodes * sizeof(double));

    if (coupling == NULL)
        return;

// Condition initiale (parallélisée)
#pragma omp parallel for
    for (i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    // Boucle temporelle
    for (step = 0; step < n_steps; ++step)
    {
//  CSR parallélisé : chaque thread traite un nœud
#pragma omp parallel for private(j, k)
        for (i = 0; i < n_nodes; ++i)
        {
            double theta_i = output[i * stride + step];
            double c_i = 0.0;

            // Parcourt les voisins de i
            for (k = row[i]; k < row[i + 1]; ++k)
            {
                j = col[k];
                double weight = edge_values[k];
                c_i += weight * sin(output[j * stride + step] - theta_i);
            }

            coupling[i] = c_i;
        }

// Mise à jour (parallélisée)
#pragma omp parallel for
        for (i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
    }

    free(coupling);
}