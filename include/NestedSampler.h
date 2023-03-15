#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "types.h"
#include <random>
#include <memory>
#include <string>
#include <set>

class Logger;

class NestedSampler {
public:
    NestedSampler(ISampler&, ILikelihood&, Logger&, NSConfig config);
    ~NestedSampler();

    void Initialise();
    void Run();
private:
    void NestedSamplingStep();
    const MCPoint SampleFromPrior();

    void UpdateLogEvidence(const double logLikelihood);
    const double EstimateLogEvidenceRemaining();

    const bool TerminateSampling();
    const double logAdd(const double logA, const double logB);
    const double logAdd(const Eigen::ArrayXd& logV);

    ISampler& mSampler;
    ILikelihood& mLikelihood;
    Logger& mLogger;
    NSConfig mConfig;

    std::multiset<MCPoint> mLivePoints;
    double mLogZ; // log evidence
//    double mLogZRemaining; // estimate of evidence remaining in live points

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniform;

    int mIter;

    const double minLikelihood = -1e30;
    const double initialLogWeight;
    const int mDimension;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

