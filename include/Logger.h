#ifndef CHMC_NESTED_SAMPLING_LOGGER_H
#define CHMC_NESTED_SAMPLING_LOGGER_H

#include "types.h"
#include <vector>
#include <string>
#include <fstream>

class Logger {
public:
    Logger(std::string name);

    virtual void WritePoint(const MCPoint&, const double logWeight);
    void WriteSummary(const NSSummary&);
    void WriteParamNames(const std::vector<std::string>& names, int totalParams);

  //  void WriteLivePoints();
  //  void ReadLivePoints();

private:
    std::string mName;
    std::string mDeadFilename;

    std::ofstream mDeadFile;
    std::ofstream mPosteriorFile;
    std::ofstream mSummaryFile;
    std::ofstream mParamNameFile;
};

#endif //CHMC_NESTED_SAMPLING_LOGGER_H
