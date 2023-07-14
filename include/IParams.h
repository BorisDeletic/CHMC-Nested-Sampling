#ifndef CHMC_NESTED_SAMPLING_IPARAMS_H
#define CHMC_NESTED_SAMPLING_IPARAMS_H

#include <Eigen/Dense>

class IParams {
public:
    virtual double GetEpsilon() const = 0;
    virtual int GetPathLength() const = 0;
    virtual const Eigen::VectorXd& GetMetric() const = 0;
};

#endif //CHMC_NESTED_SAMPLING_IPARAMS_H
