#ifndef CHMC_NESTED_SAMPLING_MOCKPARAMS_H
#define CHMC_NESTED_SAMPLING_MOCKPARAMS_H

#include "IParams.h"

class StaticParams : public IParams {
public:
    StaticParams(double epsilon, int pathLength, int dims)
    : mEpsilon(epsilon), mPathLength(pathLength), mMetric(Eigen::VectorXd::Ones(dims)) {}

    double GetEpsilon() override { return mEpsilon; };
    int GetPathLength() override { return mPathLength; };
    const Eigen::VectorXd& GetMetric() override { return mMetric; };

private:
    const double mEpsilon;
    const int mPathLength;
    const Eigen::VectorXd mMetric;
};

class MockParams : public IParams {

};

#endif //CHMC_NESTED_SAMPLING_MOCKPARAMS_H
