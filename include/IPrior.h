#ifndef CHMC_NESTED_SAMPLING_IPRIOR_H
#define CHMC_NESTED_SAMPLING_IPRIOR_H

#include <Eigen/Dense>

class IPrior {
    virtual const Eigen::VectorXd PriorTransform(const Eigen::VectorXd& cube) = 0;
};

#endif //CHMC_NESTED_SAMPLING_IPRIOR_H
