#include "kuramoto_c_kernel.h"

#include <math.h>
#include <cstdlib.h>

void simu_Rinfini(
    const double* omega,
    const double* theta0,
    double epsilon,
    double dt,
    int n_nodes,
    int n_steps,
    double* output
){
    int i;
    int j;
    int step;
    int stride =n_steps + 1;
   
    for(i=0; i< n_nodes; ++i){
        output[i*stride] = theta0[i];
    }

    for (step=0; step< n_steps; ++step){
        double c=0.0;
        double s=0.0;
        double r;
        double psi;

    for (i=0 ; i< n_nodes; ++i){
        c+= std::cos(output[i*stride + step]);
        s+= std::sin(output[i*stride + step]);
    }
    c=c/n_nodes;
    s=s/n_nodes;
    r= std::sqrt(c*c + s*s);
    if(step>n_steps - 10){
        r_somme+=r;
    }


    psi= std::atan2(s,c);
    for (j=0; j< n_nodes; ++j){
        double theta_current = output[j*stride + step];
        output[j*stride + step + 1] = theta_current + dt * (omega[j] + epsilon * r * std::sin(psi - theta_current)
            );

        }
    }
    return r_somme/10;
}