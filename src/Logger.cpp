#include "Logger.h"
#include <iostream>
#include <iomanip>

Logger::Logger(std::string name, bool logDiagnostics)
    :
    mName(name)
{
    mDeadFile.open(mName + "_dead-birth.txt");
    mPosteriorFile.open(mName + ".posterior");

    if (logDiagnostics) {
        mDiagnosticFile.open(mName + ".diagnostics");
        mLivePointsFile.open(mName + ".live_points");
        mRejectedPointsFile.open(mName + ".rejected_points");

        mDiagnosticFile
            << "iter,numlive,logZ,logZlive,likelihood,birth_likelihood,rejected,accept_prob,reflections,steps,"
            << "epsilon,path_length,metric,pdotn" << std::endl;

        mLivePointsFile << "iter,ID,likelihood,reflections,steps" << std::endl;

        mRejectedPointsFile << "accept_prob,birth_likelihood,reflections,steps,epsilon,path_length,metric" << std::endl;
    }
}


// Log weight is prior volume shell w_i = X_{i-1} - X_i
// We don't weight by L_i because we are sampling from posterior already. see 0801.3887
void Logger::WritePoint(const MCPoint& point, double logWeight) {
    if (!mDeadFile.is_open()) {
        mDeadFile.open(mName + "_dead-birth.txt", std::ios::app);
    }

    if (!mPosteriorFile.is_open()) {
        mPosteriorFile.open(mName + ".posterior", std::ios::app);
//        mPosteriorFile << std::setprecision(10);
    }

    mPosteriorFile << logWeight << " " << point.likelihood << " ";

    for (const double phi : point.derived) {
        mDeadFile << phi << " ";
        mPosteriorFile << phi << " ";
    }

    for (const double theta : point.theta) {
//        mDeadFile << theta << " ";
//        mDeadFile << std::setprecision(10) << theta << " ";
//              mPosteriorFile << theta << " ";
    }

    mDeadFile << std::setprecision(10) << point.likelihood << " " << point.birthLikelihood << std::endl;
    mPosteriorFile << std::endl;
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


void Logger::WriteSummary(const NSInfo& summary) {
    mSummaryFile.open(mName + ".stats");

    mSummaryFile << "Log (Z) = " << summary.meanLogZ << " +- " << summary.stdLogZ << std::endl;
    mSummaryFile << "Log (Z) Remaining = " << summary.logZLive << std::endl;

    mSummaryFile.close();
    mDeadFile.close();
}



void Logger::WriteLivePoints(const NSInfo& info, const std::multiset<MCPoint> &points) {
    if (!mLivePointsFile.is_open()) {
        mLivePointsFile.open(mName + ".live_points", std::ios::app);
    }

    for (const auto& point : points) {
        mLivePointsFile << info.iter << ",";
        mLivePointsFile << point.ID << ",";
        mLivePointsFile << point.likelihood << ",";
        mLivePointsFile << point.reflections << ",";
        mLivePointsFile << point.steps << std::endl;
    }
}


void Logger::WriteDiagnostics(const NSInfo& info, const MCPoint& point, const IParams& params)
{
    if (!mDiagnosticFile.is_open()) {
        mDiagnosticFile.open(mName + ".diagnostic", std::ios::app);
    }

    mDiagnosticFile << info.iter << ", ";
    mDiagnosticFile << info.numLive << ", ";
    mDiagnosticFile << info.meanLogZ << ", ";
    mDiagnosticFile << info.logZLive << ", ";

    mDiagnosticFile << point.likelihood << ", ";
    mDiagnosticFile << point.birthLikelihood << ", ";
    mDiagnosticFile << (point.rejected ? 1 : 0) << ", ";
    mDiagnosticFile << point.acceptProbability << ", ";
    mDiagnosticFile << point.reflections << ", ";
    mDiagnosticFile << point.steps << ", ";

    mDiagnosticFile << params.GetEpsilon() << ", ";
    mDiagnosticFile << params.GetPathLength() << ", ";
    mDiagnosticFile << params.GetMetric()[0];

    mDiagnosticFile << std::endl;
}


void Logger::WriteRejectedPoint(const MCPoint &point, const IParams &params) {
    if (!mRejectedPointsFile.is_open()) {
        mRejectedPointsFile.open(mName + ".rejected_points", std::ios::app);
    }

    mRejectedPointsFile << point.acceptProbability << ", ";
    mRejectedPointsFile << point.birthLikelihood << ", ";
    mRejectedPointsFile << point.reflections << ", ";
    mRejectedPointsFile << point.steps << ", ";

    mRejectedPointsFile << params.GetEpsilon() << ", ";
    mRejectedPointsFile << params.GetPathLength() << ", ";
    mRejectedPointsFile << params.GetMetric()[0] << std::endl;

    for (double dx : point.deltaX) {
        mRejectedPointsFile << dx << ", ";
    }
    mRejectedPointsFile << std::endl;

    for (double like : point.pathLikelihood) {
        mRejectedPointsFile << like << ", ";
    }
    mRejectedPointsFile << std::endl;

    for (double energy : point.pathEnergy) {
        mRejectedPointsFile << energy << ", ";
    }
    mRejectedPointsFile << std::endl;

    for (double momentum : point.pathMomentum) {
        mRejectedPointsFile << momentum << ", ";
    }
    mRejectedPointsFile << std::endl;


    mRejectedPointsFile.close();
}
