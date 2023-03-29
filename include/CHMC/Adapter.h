#ifndef CHMC_NESTED_SAMPLING_ADAPTER_H
#define CHMC_NESTED_SAMPLING_ADAPTER_H

#include "IParams.h"
#include "CHMC.h"
#include <Eigen/Dense>
#include <iostream>

class Adapter : public IParams {
public:
    Adapter(double initEpsilon, int initPathLength, const Eigen::VectorXd metric);

    double GetEpsilon() override { return mEpsilon; };
    int GetPathLength() override { return mPathLength; };
    const Eigen::VectorXd& GetMetric() override { return mMetric; };

    void Restart();
    void SetMu(double m) { mMu = m; };

    void AdaptEpsilon(double acceptProb);

private:
    double mEpsilon;
    int mPathLength;
    Eigen::VectorXd mMetric;

    int mIter;  // Adaptation iteration
    double mSBar;    // Moving average statistic
    double mXBar;    // Moving average parameter
    double mMu;       // Asymptotic mean of parameter
    double mDelta = 0.05;    // Target value of statistic
    double mGamma = 0.05;    // Adaptation scaling
    double mKappa = 0.55;    // Adaptation shrinkage
    double mT0 = 10;       // Effective starting iteration
};

#endif //CHMC_NESTED_SAMPLING_ADAPTER_H
