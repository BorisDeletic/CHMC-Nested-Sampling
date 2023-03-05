#ifndef CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H

#include "ILikelihood.h"
#include <gmock/gmock.h>

class MockLikelihood : public ILikelihood {
public:
    MOCK_METHOD(const double, LogLikelihood, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(const Eigen::VectorXd, Gradient, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(const int, GetDimension, (), (override));
};


class GaussianLikelihood : public ILikelihood {
public:
    inline GaussianLikelihood(const Eigen::ArrayXd& mean, const Eigen::ArrayXd& var) :
        mean(mean), var(var) {};


    const double LogLikelihood(const Eigen::VectorXd& theta)
    {
        double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;

        loglikelihood -= ((theta.array() - mean) / var).pow(2).sum() / 2;
        return loglikelihood;
    };

    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta)
    {
        return - ((theta.array() - mean) / var.pow(2)).matrix();
    };

    const int GetDimension() { return mean.size(); };

private:
    const Eigen::ArrayXd mean;
    const Eigen::ArrayXd var;
};



#endif //CHMC_NESTED_SAMPLING_MOCKLIKELIHOOD_H
