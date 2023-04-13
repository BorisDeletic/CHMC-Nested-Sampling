#ifndef CHMC_NESTED_SAMPLING_LOGGER_H
#define CHMC_NESTED_SAMPLING_LOGGER_H

#include "types.h"
#include <vector>
#include <string>
#include <fstream>

class Logger {
public:
    Logger(std::string name);

    void WritePoint(const MCPoint&, const Eigen::VectorXd& derivedParams);
    void WriteSummary(const NSSummary&);
    void WriteParamnames(const std::vector<std::string>& names, int totalParams);

  //  void WriteLivePoints();
  //  void ReadLivePoints();

private:
    std::string mName;
    std::string mDeadFilename;

    std::ofstream mDeadFile;
    std::ofstream mSummaryFile;
    std::ofstream mParamnameFile;
};

#endif //CHMC_NESTED_SAMPLING_LOGGER_H
