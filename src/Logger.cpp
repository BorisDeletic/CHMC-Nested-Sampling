#include "Logger.h"
#include <iostream>

Logger::Logger(std::string name)
    :
    mName(name),
    mDeadFilename(mName + "_dead-birth.txt")
{
    mDeadFile.open(mDeadFilename);
}


void Logger::WritePoint(const MCPoint& point, const Eigen::VectorXd& derivedParams) {
    if (!mDeadFile.is_open()) {
        mDeadFile.open(mDeadFilename, std::ios::app);
    }

    for (const double phi : derivedParams) {
    //    std::cout << phi << std::endl;
        mDeadFile << phi << " ";
    }

    for (const double theta : point.theta) {
        mDeadFile << theta << " ";
    }

    mDeadFile << point.likelihood << " " << point.birthLikelihood << std::endl;

}


void Logger::WriteSummary(const NSSummary& summary) {
    mSummaryFile.open(mName + ".stats");

    mSummaryFile << "Log (Z) = " << summary.logZ << std::endl;
    mSummaryFile << "Log (Z) Remaining = " << summary.logZRemaining << std::endl;

   // if (samplerSummary.rejectRatio != 0) {
   //     mSummaryFile << "Rejection Ratio = " << samplerSummary.rejectRatio << std::endl;
   // }

    mSummaryFile.close();
    mDeadFile.close();
}


void Logger::WriteParamnames(const std::vector<std::string> &names, int totalParams)
{
    mParamnameFile.open(mName + ".paramnames");

    for (const auto& name : names) {
        mParamnameFile << name[0] << " " << name << std::endl;
    }

    for (int i = 1; i < totalParams - names.size() + 1; i++) {
        mParamnameFile << "p" << i << " \\theta{" << i << "}\n";
    }

    mParamnameFile.close();
}
