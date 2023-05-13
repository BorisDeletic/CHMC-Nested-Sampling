//
// Created by Boris Deletic on 11/02/2023.
//

#ifndef CHMC_NESTED_SAMPLING_LIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_LIKELIHOOD_H

namespace Likelihood
{
    double likelihood(double* theta, int size, double A);
    double gradient(double* theta, double* d_theta, int size, double A);
};

//double likelihood(double* theta, int n);
//double gradient(double* theta, double* d_theta, int size);


#endif //CHMC_NESTED_SAMPLING_LIKELIHOOD_H
