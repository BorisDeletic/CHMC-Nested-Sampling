#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "Adapter.h"
#include "types.h"
#include <random>
#include <memory>
#include <string>
#include <set>

class Logger;

class NestedSampler {
public:
    NestedSampler(ISampler&, ILikelihood&, Logger&, NSConfig config);

    void SetAdaption(Adapter* adapter);
    void Initialise();
    void Run();

private:
    void NestedSamplingStep();
    void SampleNewPoint(const MCPoint& deadPoint, const double likelihoodConstraint);
    const MCPoint SampleFromPrior();
    const MCPoint& GetRandomPoint();

    void UpdateLogEvidence(double logLikelihood);
    const double EstimateLogEvidenceRemaining();

    const bool TerminateSampling();
    const double logAdd(double logA, double logB);
    const double logAdd(const Eigen::ArrayXd& logV);

    ISampler& mSampler;
    ILikelihood& mLikelihood;
    Logger& mLogger;
    Adapter* mAdapter = nullptr;

    NSConfig mConfig;

    std::multiset<MCPoint> mLivePoints;
    double mLogZ; // log evidence
//    double mLogZRemaining; // estimate of evidence remaining in live points

    int mIter;
    int mReflections;
    int mIntegrationSteps;

    const double minLikelihood = -1e30;
    const double initialLogWeight;
    const int mDimension;
    const int mSampleRetries = 5;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniform;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

