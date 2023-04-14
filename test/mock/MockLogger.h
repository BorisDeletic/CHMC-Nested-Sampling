#ifndef CHMC_NESTED_SAMPLING_MOCKLOGGER_H
#define CHMC_NESTED_SAMPLING_MOCKLOGGER_H

#include "Logger.h"
#include <gmock/gmock.h>

class MockLogger : public Logger {
public:
    MockLogger() : Logger("MockLogger") {}
    MOCK_METHOD(void, WritePoint, (const MCPoint&, const Eigen::VectorXd& derivedParams), (override));
};

#endif //CHMC_NESTED_SAMPLING_MOCKLOGGER_H
