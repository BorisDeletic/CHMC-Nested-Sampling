#ifndef CHMC_NESTED_SAMPLING_MOCKLOGGER_H
#define CHMC_NESTED_SAMPLING_MOCKLOGGER_H

#include "Logger.h"
#include <gmock/gmock.h>

class MockLogger : public Logger {
public:
    MOCK_METHOD(void, WriteDeadPoint, (const MCPoint&), (override));
};

#endif //CHMC_NESTED_SAMPLING_MOCKLOGGER_H
