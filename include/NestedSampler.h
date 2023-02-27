#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include "CHMC.h"
#include "IPrior.h"
#include "types.h"
#include <memory>
#include <string>
#include <set>

class Logger;

class NestedSampler {
public:
    NestedSampler(CHMC&, IPrior&, ILikelihood&, int numLive, std::string name);
    ~NestedSampler();

    void Initialise();
    void Run(int steps);
private:
    void NestedSamplingStep();
    const MCPoint SampleFromPrior();

    CHMC& mCHMC;
    IPrior& mPrior;
    ILikelihood& mLikelihood;
    std::unique_ptr<Logger> mLogger;

    std::multiset<MCPoint> mLivePoints;
    const int mNumLive;
    const int mDimension;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniform;

    std::string mName;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

