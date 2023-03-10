//
// Created by Boris Deletic on 11/02/2023.
//

#include "Likelihood.h"

//log likelihood
double Likelihood::likelihood(double* theta, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += theta[i] * theta[i] * theta[i];
    }
    return sum;
}


int enzyme_dup;
int enzyme_const;
extern double __enzyme_autodiff(...);
double Likelihood::gradient(double* theta, double* d_theta, int size) {
    return __enzyme_autodiff(likelihood,
                             enzyme_dup, theta, d_theta,
                             enzyme_const, size);
}
// d_theta must be allocated and initialised to zero.
