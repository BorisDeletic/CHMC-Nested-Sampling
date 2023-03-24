#ifndef CHMC_NESTED_SAMPLING_IPARAMS_H
#define CHMC_NESTED_SAMPLING_IPARAMS_H

#include <Eigen/Dense>

class IParams {
public:
    virtual double GetEpsilon() = 0;
    virtual int GetPathLength() = 0;
    virtual const Eigen::VectorXd& GetMetric() = 0;
};

#endif //CHMC_NESTED_SAMPLING_IPARAMS_H
