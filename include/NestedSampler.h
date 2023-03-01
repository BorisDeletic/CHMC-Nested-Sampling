#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "IPrior.h"
#include "types.h"
#include <random>
#include <memory>
#include <string>
#include <set>

class Logger;

class NestedSampler {
public:
    NestedSampler(ISampler&, IPrior&, ILikelihood&, Logger&, int numLive);
    ~NestedSampler();

    void Initialise();
    void Run(int steps);
private:
    void NestedSamplingStep();
    const MCPoint SampleFromPrior();

    ISampler& mSampler;
    IPrior& mPrior;
    ILikelihood& mLikelihood;
    Logger& mLogger;

    std::multiset<MCPoint> mLivePoints;
    const int mNumLive;
    const int mDimension;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniform;

    const double maxLikelihood = -1e30;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

