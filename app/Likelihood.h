//
// Created by Boris Deletic on 11/02/2023.
//
#include "ILikelihood.h"

#ifndef CHMC_NESTED_SAMPLING_LIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_LIKELIHOOD_H

class Likelihood : public ILikelihood {
    double likelihood(double *theta, int n) override;
    double gradient(double* theta, double* d_theta, int size) override;
};

#endif //CHMC_NESTED_SAMPLING_LIKELIHOOD_H
