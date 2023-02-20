#ifndef CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_ILIKELIHOOD_H

#include <Eigen/Dense>

class ILikelihood {
public:
    virtual double Likelihood(const Eigen::VectorXd& x) = 0;
    virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& x) = 0;
};

#endif //CHMC_NESTED_SAMPLING_ILIKELIHOOD_H
