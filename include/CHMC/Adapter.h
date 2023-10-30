#ifndef CHMC_NESTED_SAMPLING_ADAPTER_H
#define CHMC_NESTED_SAMPLING_ADAPTER_H

#include "IParams.h"
#include "types.h"
#include <Eigen/Dense>
#include <iostream>
#include <set>

class Adapter : public IParams {
public:
    Adapter(int dimension, double initEpsilon, int initPathLength, double reflectRateTarget);

    double GetEpsilon() const override { return mEpsilon; };
    int GetPathLength() const override { return mPathLength; };
    const Eigen::VectorXd& GetMetric() const override { return mMetric; };

    void Restart();
    void SetMu(double m) { mMu = m; };

    void AdaptEpsilon(double acceptProb);
    void AdaptMetric(const std::multiset<MCPoint>& livePoints);

private:

    double mEpsilon;
    int mPathLength;
    int mDimension;
    Eigen::VectorXd mMetric;

    int mIter;  // Adaptation iteration
    double mSBar;    // Moving average statistic
    double mXBar;    // Moving average parameter
    double mMu;       // Asymptotic mean of parameter
    double mDelta = 0.01;    // Target value of statistic
    double mGamma = 0.05;    // Adaptation scaling
    double mKappa = 0.75;    // Adaptation shrinkage
    double mT0 = 10;       // Effective starting iteration

    double mAlpha = 0.1;
};

#endif //CHMC_NESTED_SAMPLING_ADAPTER_H
