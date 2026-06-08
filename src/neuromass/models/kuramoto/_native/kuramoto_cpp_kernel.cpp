#include "kuramoto_cpp_kernel.hpp"

#include <cmath>

void simulate_naive_kuramoto_cpp(
    const double* adjacency,
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
) {
    int stride = n_steps + 1;

    for (int i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }

    for (int step = 0; step < n_steps; ++step) {
        for (int i = 0; i < n_nodes; ++i) {
            double coupling = 0.0;
            for (int j = 0; j < n_nodes; ++j) {
                double weight = adjacency[i * n_nodes + j];
                if (weight != 0.0) {
                    coupling += weight * std::sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (
                omega[i] + (epsilon / n_nodes) * coupling
            );
        }
    }
}

