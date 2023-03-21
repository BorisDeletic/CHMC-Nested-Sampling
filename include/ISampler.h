#ifndef CHMC_NESTED_SAMPLING_ISAMPLER_H
#define CHMC_NESTED_SAMPLING_ISAMPLER_H

#include "types.h"

class ISampler {
public:
    virtual void Initialise(const MCPoint& init) = 0;
    virtual const MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint) = 0;
    virtual const SamplerSummary GetSummary() = 0;
};

#endif //CHMC_NESTED_SAMPLING_ISAMPLER_H
