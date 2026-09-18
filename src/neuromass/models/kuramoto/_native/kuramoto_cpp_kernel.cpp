#include "kuramoto_cpp_kernel.hpp"
#include <omp.h>
#include <cmath>

// VERSION 1 : NAIVE (DENSE) - SEQUENTIEL

void simulate_naive_kuramoto_cpp(
    const double *adjacency,
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double *output)
{
    int stride = n_steps + 1;

    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (int step = 0; step < n_steps; ++step)
    {
        for (int i = 0; i < n_nodes; ++i)
        {
            double coupling = 0.0;
            for (int j = 0; j < n_nodes; ++j)
            {
                double weight = adjacency[i * n_nodes + j];
                if (weight != 0.0)
                {
                    coupling += weight * std::sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling);
        }
    }
}

// VERSION 2 : MEAN-FIELD - SEQUENTIEL

void simu_para_complexe_cpp(
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double *output)
{
    int stride = n_steps + 1;

    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (int step = 0; step < n_steps; ++step)
    {
        double c = 0.0;
        double s = 0.0;

        for (int i = 0; i < n_nodes; ++i)
        {
            c += std::cos(output[i * stride + step]);
            s += std::sin(output[i * stride + step]);
        }
        c = c / n_nodes;
        s = s / n_nodes;
        double r = std::sqrt(c * c + s * s);
        double psi = std::atan2(s, c);

        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + epsilon * r * std::sin(psi - output[i * stride + step]));
        }
    }
}

// VERSION 3 : SPARSE - SEQUENTIEL

void simu_sparse_cpp(
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
    int stride = n_steps + 1;
    double *coupling = new double[n_nodes];

    // Condition initiale
    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    // Boucle temporelle
    for (int step = 0; step < n_steps; ++step)
    {
        // CSR : parcourt les voisins de chaque nœud
        for (int i = 0; i < n_nodes; ++i)
        {
            double theta_i = output[i * stride + step];
            double c_i = 0.0;

            // Parcourt les voisins de i
            for (int k = row[i]; k < row[i + 1]; ++k)
            {
                int j = col[k];
                double weight = edge_values[k];
                c_i += weight * std::sin(output[j * stride + step] - theta_i);
            }

            coupling[i] = c_i;
        }

        // Mise à jour
        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
    }

    delete[] coupling;
}

// VERSION 4 : NAIVE (DENSE) - OPENMP

void simulate_naive_kuramoto_cpp_omp(
    const double *adjacency,
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double *output)
{
    int stride = n_steps + 1;

#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (int step = 0; step < n_steps; ++step)
    {
#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            double coupling = 0.0;
            for (int j = 0; j < n_nodes; ++j)
            {
                double weight = adjacency[i * n_nodes + j];
                if (weight != 0.0)
                {
                    coupling += weight * std::sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling);
        }
    }
}

// VERSION 5 : MEAN-FIELD - OPENMP

void simu_para_complexe_cpp_omp(
    const double *omega,
    const double *theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double *output)
{
    int stride = n_steps + 1;

#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    for (int step = 0; step < n_steps; ++step)
    {
        double c = 0.0;
        double s = 0.0;

#pragma omp parallel for reduction(+ : c, s)
        for (int i = 0; i < n_nodes; ++i)
        {
            c += std::cos(output[i * stride + step]);
            s += std::sin(output[i * stride + step]);
        }
        c = c / n_nodes;
        s = s / n_nodes;
        double r = std::sqrt(c * c + s * s);
        double psi = std::atan2(s, c);

#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + epsilon * r * std::sin(psi - output[i * stride + step]));
        }
    }
}

// VERSION 6 : SPARSE - OPENMP

void simu_sparse_cpp_omp(
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
    int stride = n_steps + 1;
    double *coupling = new double[n_nodes];

// Condition initiale (parallélisée)
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }

    // Boucle temporelle
    for (int step = 0; step < n_steps; ++step)
    {
// CSR parallélisé : chaque thread traite un nœud
#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            double theta_i = output[i * stride + step];
            double c_i = 0.0;

            // Parcourt les voisins de i
            for (int k = row[i]; k < row[i + 1]; ++k)
            {
                int j = col[k];
                double weight = edge_values[k];
                c_i += weight * std::sin(output[j * stride + step] - theta_i);
            }

            coupling[i] = c_i;
        }

// Mise à jour (parallélisée)
#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
    }

    delete[] coupling;
}
