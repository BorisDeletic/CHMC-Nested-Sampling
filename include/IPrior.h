#ifndef CHMC_NESTED_SAMPLING_IPRIOR_H
#define CHMC_NESTED_SAMPLING_IPRIOR_H

#include <Eigen/Dense>

class IPrior {
public:
    virtual const Eigen::VectorXd PriorTransform(const Eigen::VectorXd& cube) = 0;
    virtual const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) = 0;

    virtual const int GetDimension() = 0;
};

#endif //CHMC_NESTED_SAMPLING_IPRIOR_H
