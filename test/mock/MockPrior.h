#ifndef CHMC_NESTED_SAMPLING_MOCKPRIOR_H
#define CHMC_NESTED_SAMPLING_MOCKPRIOR_H

#include "IPrior.h"
#include <gmock/gmock.h>

class MockPrior : public IPrior {
public:
    MOCK_METHOD(const Eigen::VectorXd, PriorTransform, (const Eigen::VectorXd&), (override));
    MOCK_METHOD(const int, GetDimension, (), (override));
};


//class GaussianLogLikelihood : public ILikelihood {
//public:
//    inline GaussianLogLikelihood(const Eigen::ArrayXd& mean, const Eigen::ArrayXd& var) :
//            mean(mean), var(var) {};
//
//
//    double Likelihood(const Eigen::VectorXd& theta)
//    {
//        double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;
//
//        loglikelihood -= ((theta.array() - mean) / var).pow(2).sum() / 2;
//        return loglikelihood;
//    };
//
//    Eigen::VectorXd Gradient(const Eigen::VectorXd& theta)
//    {
//        return - ((theta.array() - mean) / var.pow(2)).matrix();
//    };
//
//    const int GetDimension() { return mean.size(); };
//
//private:
//    const Eigen::ArrayXd mean;
//    const Eigen::ArrayXd var;
//};
//

#endif //CHMC_NESTED_SAMPLING_MOCKPRIOR_H
