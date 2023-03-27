#ifndef CHMC_NESTED_SAMPLING_MOCKSAMPLER_H
#define CHMC_NESTED_SAMPLING_MOCKSAMPLER_H

#include "ISampler.h"
#include <gmock/gmock.h>

class MockSampler : public ISampler {
public:
    MOCK_METHOD(const MCPoint, SamplePoint, (const MCPoint&, double), (override));
};

#endif //CHMC_NESTED_SAMPLING_MOCKSAMPLER_H
