#include "NestedSampler.h"
#include "Logger.h"


NestedSampler::NestedSampler(std::string name)
    : mName(name)
{
    mLogger = std::make_unique<Logger>(mName);
}



