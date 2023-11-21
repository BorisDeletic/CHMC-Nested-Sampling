#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include "ISampler.h"
#include "IPrior.h"
#include "ILikelihood.h"
#include "Adapter.h"
#include "types.h"
#include <random>
#include <memory>
#include <string>
#include <set>
#include <cfloat>

class Logger;

class NestedSampler {
public:
    NestedSampler(ISampler&, IPrior&, ILikelihood&, Logger&, NSConfig config);

    void SetAdaption(Adapter* adapter);
    void Initialise();
    void Run();

    const NSInfo GetInfo();
private:
    void NestedSamplingStep();
    void SampleNewPoint(const MCPoint& deadPoint, double likelihoodConstraint);
    const MCPoint SampleFromPrior();
    const MCPoint& GetRandomLivePoint();

    void UpdateLogEvidence(const MCPoint&);
    const double EstimateLogEvidenceRemaining();
    const double GetReflectRate();

    const bool TerminateSampling();

    const double logAdd(double logA, double logB);
    const double logAdd(const Eigen::ArrayXd& logV);

    ISampler& mSampler;
    IPrior& mPrior;
    ILikelihood& mLikelihood;
    Logger& mLogger;
    Adapter* mAdapter = nullptr;

    const NSConfig mConfig;

    std::multiset<MCPoint> mLivePoints;
//    double mLogZRemaining; // estimate of evidence remaining in live points

    int mIter;

    double mLogZ =  -DBL_MAX; // log evidence. Z_init = 0
    double mLogZZ = -DBL_MAX; // log Z^2. init = 0
    double mLogZX = -DBL_MAX; // log ZX.  init = 0
    double mLogX = 0; // log prior volume. X_init = 1
    double mLogXX = 0; // log XX. init = 1

    const double minLikelihood = -1e30;
    const int mDimension;
    const int mSampleRetries = 5;

    int rejections = 0;
    int samples = 0;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniformRng;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

