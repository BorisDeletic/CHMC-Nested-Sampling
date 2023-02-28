#include "Logger.h"

Logger::Logger(std::string fname)
    : mFilename(fname)
{
    mFile.open(mFilename);
}


void Logger::WriteDeadPoint(const MCPoint& point) {
    if (!mFile.is_open()) {
        mFile.open(mFilename, std::ios::app);
    }

    for (const double x : point.theta) {
        mFile << x << " ";
    }

    mFile << point.likelihood << " " << point.birthLikelihood << std::endl;
}