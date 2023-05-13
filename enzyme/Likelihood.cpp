//
// Created by Boris Deletic on 11/02/2023.
//

#include "Likelihood.h"
#include <cmath>

//Rosenbrock Likelihood
double Likelihood::likelihood(double* theta, int size, double A) {
    double f = A * size;

    for (int i = 0; i < size; i++) {
        f += pow(theta[i], 2) - A * cos(2 * M_PI * theta[i]);
    }

    return f;
}


int enzyme_dup;
int enzyme_const;
extern double __enzyme_autodiff(...);
double Likelihood::gradient(double* theta, double* d_theta, int size, double A) {
    return __enzyme_autodiff(likelihood,
                             enzyme_dup, theta, d_theta,
                             enzyme_const, size, A);
}
// d_theta must be allocated and initialised to zero.
