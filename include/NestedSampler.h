#ifndef CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H
#define CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

#include <memory>
#include <string>
#include <set>

class Logger;

class NestedSampler {
public:
    NestedSampler(std::string name);

private:
    std::unique_ptr<Logger> mLogger;
    std::set

    std::string mName;
};

#endif //CHMC_NESTED_SAMPLING_NESTEDSAMPLING_H

