#include "kuramoto_c_kernel.h"

#include <math.h>

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
    int i;
    int j;
    int step;
    int stride = n_steps + 1;
    double coupling;
    double weight;

    for (i = 0; i < n_nodes; ++i) {
        output[i * stride] = theta0[i];
    }

    for (step = 0; step < n_steps; ++step) {
        for (i = 0; i < n_nodes; ++i) {
            coupling = 0.0;
            for (j = 0; j < n_nodes; ++j) {
                weight = adjacency[i * n_nodes + j];
                if (weight != 0.0) {
                    coupling += weight * sin(output[j * stride + step] - output[i * stride + step]);
                }
            }
            output[i * stride + step + 1] = output[i * stride + step] + dt * (
                omega[i] + (epsilon / n_nodes) * coupling
            );
        }
    }
}
