#ifndef CHMC_NESTED_SAMPLING_METROPOLIS_H
#define CHMC_NESTED_SAMPLING_METROPOLIS_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "types.h"
#include <random>

class RejectionSampler : public ISampler {
public:
    RejectionSampler(ILikelihood&, double epsilon);

    const MCPoint SamplePoint(const MCPoint& old, const double likelihoodConstraint);

private:
    ILikelihood& mLikelihood;

    const double mEpsilon;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> mUniform;
};

#endif //CHMC_NESTED_SAMPLING_METROPOLIS_H
