#ifndef CHMC_NESTED_SAMPLING_LOGGER_H
#define CHMC_NESTED_SAMPLING_LOGGER_H

#include "types.h"
#include <string>
#include <fstream>

class Logger {
public:
    Logger(std::string name);

    void WritePoint(const MCPoint&, const Eigen::VectorXd& derivedParams);
    void WriteSummary(const NSSummary&);

  //  void WriteLivePoints();
  //  void ReadLivePoints();

private:
    std::string mName;
    std::string mDeadFilename;
    std::string mSummaryFilename;

    std::ofstream mDeadFile;
    std::ofstream mSummaryFile;
};

#endif //CHMC_NESTED_SAMPLING_LOGGER_H
