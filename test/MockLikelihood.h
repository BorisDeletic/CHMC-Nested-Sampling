#ifndef CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H

#include "ILikelihood.h"
#include <gmock/gmock.h>

class MockLikelihood : public ILikelihood {
public:
    MOCK_METHOD(double, Likelihood, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(Eigen::VectorXd, Gradient, (const Eigen::VectorXd&), (override));
};


class GaussianLogLikelihood : public ILikelihood {
public:
    inline GaussianLogLikelihood(const Eigen::ArrayXd& mean, const Eigen::ArrayXd& var) :
        mean(mean), var(var) {};


    double Likelihood(const Eigen::VectorXd& theta)
    {
        double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;

        loglikelihood -= ((theta.array() - mean) / var).pow(2).sum() / 2;
        return loglikelihood;
    };

    Eigen::VectorXd Gradient(const Eigen::VectorXd& theta)
    {
        return - ((theta.array() - mean) / var.pow(2)).matrix();
    };

private:
    const Eigen::ArrayXd mean;
    const Eigen::ArrayXd var;
};



#endif //CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
