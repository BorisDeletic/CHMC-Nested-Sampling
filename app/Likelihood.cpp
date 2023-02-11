//
// Created by Boris Deletic on 11/02/2023.
//

#include "ILikelihood.h"

int enzyme_dup;
int enzyme_const;
extern double __enzyme_autodiff(...);

class Likelihood : public ILikelihood {
    double likelihood(double *theta, int n) override {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += theta[i] * theta[i] * theta[i];
        }
        return sum;
    }

    double gradient(double* theta, double* d_theta, int size) {
        double (ILikelihood::*func)(double*, int);
        func = &ILikelihood::likelihood;
        // This returns the derivative of square or 2 * x
        return __enzyme_autodiff(func,
                                 enzyme_dup, theta, d_theta,
                                 enzyme_const, size);
    }

};
