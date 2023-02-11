//
// Created by Boris Deletic on 11/02/2023.
//
// Likelihood Interface class

#ifndef CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_ILIKELIHOOD_H

class ILikelihood
{
public:
    virtual double likelihood(double* theta, int n) = 0;
    virtual double gradient(double* theta, double* d_theta, int size) = 0;
};

#endif //CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
