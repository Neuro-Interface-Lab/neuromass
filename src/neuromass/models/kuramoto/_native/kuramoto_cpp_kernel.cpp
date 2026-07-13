#include "kuramoto_cpp_kernel.hpp"
#include <omp.h>
#include <cmath>

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
void simu_para_complexe(
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
        double r, psi;

#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            c += std::cos(output[i * stride + step]);
            s += std::sin(output[i * stride + step]);
        }

        r = std::sqrt(c * c + s * s) / n_nodes;
        psi = std::atan2(s, c);

        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + epsilon * r * std::sin(psi - output[i * stride + step]));
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

    int stride = n_steps + 1;
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i)
    {
        output[i * stride] = theta0[i];
    }
    for (int step = 0; step < n_steps; ++step)
    {
        double *coupling = new double[n_nodes]();
        for (int e = 0; e < n_edges; ++e)
        {
            int i = edge_rows[e];
            int j = edge_cols[e];
            double weight = edge_values[e];
            double theta_i = output[i * stride + step];
            double theta_j = output[j * stride + step];
            coupling[i] += weight * std::sin(theta_j - theta_i);
        }
#pragma omp parallel for
        for (int i = 0; i < n_nodes; ++i)
        {
            output[i * stride + step + 1] = output[i * stride + step] + dt * (omega[i] + (epsilon / n_nodes) * coupling[i]);
        }
        delete[] coupling;
    }
}
