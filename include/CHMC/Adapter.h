#ifndef CHMC_NESTED_SAMPLING_ADAPTER_H
#define CHMC_NESTED_SAMPLING_ADAPTER_H

#include "IParams.h"
#include "CHMC.h"
#include <Eigen/Dense>

class Adapter : public IParams {
public:
    Adapter(CHMC& chmc, double initEpsilon, int initPathLength, const Eigen::VectorXd metric);

    double GetEpsilon() override { return mEpsilon; };
    int GetPathLength() override { return mPathLength; };
    const Eigen::VectorXd& GetMetric() override { return mMetric; };

private:
    CHMC& mCHMC;

    double mEpsilon;
    int mPathLength;
    Eigen::VectorXd mMetric;
};

#endif //CHMC_NESTED_SAMPLING_ADAPTER_H
