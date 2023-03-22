#include "Logger.h"

Logger::Logger(std::string name)
    :
    mName(name),
    mDeadFilename(name + "_dead-birth.txt"),
    mSummaryFilename(mName + ".stats")
{
    mDeadFile.open(mDeadFilename);
}


void Logger::WritePoint(const MCPoint& point) {
    if (!mDeadFile.is_open()) {
        mDeadFile.open(mDeadFilename, std::ios::app);
    }

    for (const double x : point.theta) {
        mDeadFile << x << " ";
    }

    mDeadFile << point.likelihood << " " << point.birthLikelihood << std::endl;
}


void Logger::WriteSummary(const NSSummary& summary, const SamplerSummary &samplerSummary) {
    mSummaryFile.open(mSummaryFilename);

    mSummaryFile << "Log (Z) = " << summary.logZ << std::endl;
    mSummaryFile << "Log (Z) Remaining = " << summary.logZRemaining << std::endl;

    if (samplerSummary.rejectRatio != 0) {
        mSummaryFile << "Rejection Ratio = " << samplerSummary.rejectRatio << std::endl;
    }

    mSummaryFile.close();
}
