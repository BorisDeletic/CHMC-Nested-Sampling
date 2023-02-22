#ifndef CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H

#include "ILikelihood.h"
#include <gmock/gmock.h>

class MockLikelihood : public ILikelihood {
public:
    MOCK_METHOD(double, Likelihood, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(Eigen::VectorXd, Gradient, (const Eigen::VectorXd&), (override));
};


const double GaussianLogLikelihood(const Eigen::ArrayXd& theta, const Eigen::ArrayXd& mean, const Eigen::ArrayXd& var) {
    double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;

    loglikelihood -= ((theta - mean) / var).pow(2).sum() / 2;

    return loglikelihood;
}

Eigen::VectorXd GaussianGradient(const Eigen::ArrayXd& theta, const Eigen::ArrayXd& mean, const Eigen::ArrayXd& var) {
    return ((theta - mean) / var.pow(2)).matrix();
}


#endif //CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
