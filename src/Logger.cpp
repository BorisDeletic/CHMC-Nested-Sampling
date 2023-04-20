#include "Logger.h"
#include <iostream>

Logger::Logger(std::string name)
    :
    mName(name),
    mDeadFilename(mName + "_dead-birth.txt")
{
    mDeadFile.open(mDeadFilename);
    mPosteriorFile.open(mName + ".posterior");
}


void Logger::WritePoint(const MCPoint& point, const double logWeight) {
    if (!mDeadFile.is_open()) {
        mDeadFile.open(mDeadFilename, std::ios::app);
    }

    if (!mPosteriorFile.is_open()) {
        mPosteriorFile.open(mName + ".posterior", std::ios::app);
    }

    double posteriorWeight = exp(logWeight + point.likelihood);
    mPosteriorFile << posteriorWeight << " ";
    mPosteriorFile << -point.likelihood << " ";

    for (const double phi : point.derived) {
        mDeadFile << phi << " ";
        mPosteriorFile << phi << " ";
    }

    for (const double theta : point.theta) {
        mDeadFile << theta << " ";
  //      mPosteriorFile << theta << " ";
    }

    mDeadFile << point.likelihood << " " << point.birthLikelihood << std::endl;
    mPosteriorFile << std::endl;
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


void Logger::WriteParamNames(const std::vector<std::string> &names, int totalParams)
{
    mParamNameFile.open(mName + ".paramnames");

    for (const auto& name : names) {
        mParamNameFile << name << " " << name << std::endl;
    }

    for (int i = 1; i < totalParams - names.size() + 1; i++) {
        mParamNameFile << "p" << i << " \\theta{" << i << "}\n";
    }

    mParamNameFile.close();
}
