#ifndef CHMC_NESTED_SAMPLING_LOGGER_H
#define CHMC_NESTED_SAMPLING_LOGGER_H

#include "types.h"
#include <string>
#include <fstream>

class Logger {
public:
    Logger(std::string fname);

    void WriteDeadPoint(const MCPoint& point);
  //  void WriteLivePoints();
  //  void ReadLivePoints();

private:
    std::string mFilename;
    std::ofstream mFile;
};

#endif //CHMC_NESTED_SAMPLING_LOGGER_H
