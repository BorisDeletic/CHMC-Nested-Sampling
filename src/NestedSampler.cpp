#include "NestedSampler.h"
#include "Logger.h"
#include <iostream>
#include <cassert>
#include <exception>


NestedSampler::NestedSampler(ISampler& sampler, IPrior& prior, ILikelihood& likelihood, Logger& logger,
                             NSConfig config)
    :
        mSampler(sampler),
        mPrior(prior),
        mLikelihood(likelihood),
        mLogger(logger),
        mConfig(config),
        mDimension(mLikelihood.GetDimension()),
        gen(rd()),
        mUniformRng(0, 1)
{
    if (mPrior.GetDimension() != mLikelihood.GetDimension()) {
        throw std::runtime_error("Prior dimension and likelihood dimensions do not match");
    }
}


void NestedSampler::Initialise() {
    mIter = 0;
    mLivePoints.clear();

    for (int i = mLivePoints.size(); i < mConfig.numLive; i++) {
        MCPoint newPoint = SampleFromPrior();
        mLivePoints.insert(newPoint);
    }

}


void NestedSampler::SetAdaption(Adapter* adapter) {
    mAdapter = adapter;
}


void NestedSampler::Run() {
    bool terminationCondition = false;
    while (!terminationCondition)
    {
        NestedSamplingStep();
        mIter++;

        if (mIter % mConfig.numLive == 0) {
            terminationCondition = TerminateSampling();
        }
    }

    // log all remaining live points
    for (const auto& point : mLivePoints)
    {
        UpdateLogEvidence(point);
        mLogger.WritePoint(point);
    }

    mLogger.WriteSummary(GetInfo());

    int totalParams = mDimension + mLikelihood.DerivedParams(Eigen::VectorXd::Ones(mDimension)).size();
    mLogger.WriteParamNames(mLikelihood.ParamNames(), totalParams);
}


void NestedSampler::NestedSamplingStep() {
    auto lowestIt = mLivePoints.begin();
    const MCPoint &deadPoint = *lowestIt; // lowest likelihood point

    // Analysis and log dead point
    UpdateLogEvidence(deadPoint);
    mLogger.WritePoint(deadPoint);

    const MCPoint& randPoint = GetRandomLivePoint();
    SampleNewPoint(randPoint, deadPoint.likelihood);

    //kill point.
    mLivePoints.erase(lowestIt);
}


void NestedSampler::SampleNewPoint(const MCPoint& deadPoint, const double likelihoodConstraint)
{
    for (int i = 0; i < mSampleRetries; i++) {
        const MCPoint newPoint = mSampler.SamplePoint(deadPoint, likelihoodConstraint);

        if (mAdapter != nullptr)
        {
            const double reflectRate = (double)newPoint.reflections / newPoint.steps;
            mAdapter->AdaptEpsilon(reflectRate);
        }

        if (mConfig.logDiagnostics) {
            assert(mAdapter != nullptr);
            mLogger.WriteDiagnostics(GetInfo(), newPoint, *mAdapter);
        }

        if (!newPoint.rejected)
        {
            // sampled valid new point
            mLivePoints.insert(newPoint);

            if (mLivePoints.size() > mConfig.numLive)
                return;
        }

//        if (mConfig.logDiagnostics && newPoint.rejected) {
        if (newPoint.rejected) {
            std::cout << "rejected" << std::endl;
//            std::cout << "e = " << mAdapter->GetEpsilon() << std::endl;
            mLogger.WriteRejectedPoint(newPoint, *mAdapter);
        }
    }
}


const MCPoint NestedSampler::SampleFromPrior() {
    Eigen::VectorXd cube = Eigen::VectorXd::NullaryExpr(mDimension, [&](){
        return mUniformRng(gen);
    });
    Eigen::VectorXd theta = mPrior.PriorTransform(cube);

    MCPoint pointFromPrior = {
            theta,
            mLikelihood.DerivedParams(theta),
            mLikelihood.LogLikelihood(theta),
            minLikelihood
    };

    return pointFromPrior;
}


// see arxiv 1506.00171 appendix B
void NestedSampler::UpdateLogEvidence(const MCPoint& point) {
    int n = mConfig.numLive;

    // global evidence
    double dLogZ = mLogX + point.likelihood - log(n + 1);
    mLogZ = logAdd(mLogZ, dLogZ);

    // compress prior volume
    mLogX = mLogX + log(n) - log(n + 1);

    // evidence error
    double dLogZZ_term1 = point.likelihood + mLogZX + log(2) - log(n+1);
    double dLogZZ_term2 = 2 * point.likelihood + mLogXX - log(n+1) - log(n+2);
    double dLogZZ = logAdd(dLogZZ_term1, dLogZZ_term2);
    mLogZZ = logAdd(mLogZZ, dLogZZ);

    // evidence volume correlation
    double dLogZX = mLogXX + point.likelihood + log(n) - log(n + 1) - log(n + 2);
    mLogZX = mLogZX + log(n) - log(n + 1);
    mLogZX = logAdd(mLogZX, dLogZX);

    // volume correlation
    mLogXX = mLogXX + log(n) - log(n + 2);
}


// Estimate Log evidence remaining in live points
const double NestedSampler::EstimateLogEvidenceRemaining() {
    Eigen::ArrayXd logLikelihoodLive(mLivePoints.size());

    int i = 0;
    for (auto& point : mLivePoints) {
        logLikelihoodLive[i] = point.likelihood;
        i++;
    }

    double logMeanLiveLikelihood = logAdd(logLikelihoodLive) - log(mConfig.numLive);

    double logEvidenceLive = logMeanLiveLikelihood - (double)mIter / mConfig.numLive;
    return logEvidenceLive;
}


const MCPoint& NestedSampler::GetRandomLivePoint() {
    const int randomIndex = std::floor(mUniformRng(gen) * mLivePoints.size());

    auto It = mLivePoints.begin();
    std::advance(It, randomIndex);

    return *It;
}


const double NestedSampler::GetReflectRate() {
    double reflections = 0;
    double steps = 0;

    for (auto& point : mLivePoints) {
        reflections += point.reflections;
        steps += point.steps;
    }

    return 100 * reflections / steps;
}


const NSInfo NestedSampler::GetInfo() {

    double meanLogZ = 2 * mLogZ - 0.5 * mLogZZ;
    double stdLogZ  = sqrt(mLogZZ - 2 * mLogZ);

    NSInfo info = {
            mIter,
            mConfig.numLive,
            meanLogZ,
            stdLogZ,
            EstimateLogEvidenceRemaining()
    };

    return info;
}


const bool NestedSampler::TerminateSampling() {
    if (mAdapter != nullptr)
    {
        mAdapter->AdaptMetric(mLivePoints);

        std::cout << "NS Step: " << mIter << ", Num Live = " << mLivePoints.size() << std::endl;
        std::cout << "e=" << mAdapter->GetEpsilon() << ", reflectionrate=" << GetReflectRate() << std::endl;
        std::cout << "alpha=" << mAdapter->GetMetric()[0] << std::endl;
    }

    double remainingEvidence = EstimateLogEvidenceRemaining();
    std::cout << "Step: " << mIter << std::endl;
    std::cout << "Log(Z)=" << mLogZ << " ,LogZlive=" << remainingEvidence << std::endl;

    double meanLogZ = 2 * mLogZ - 0.5 * mLogZZ;
    double stdLogZ  = sqrt(mLogZZ - 2 * mLogZ);
    std::cout << "Normal Log(Z)=" << meanLogZ << " +- " << stdLogZ << std::endl << std::endl;


    if (mConfig.logDiagnostics) {
        mLogger.WriteLivePoints(GetInfo(), mLivePoints);
    }

    if (remainingEvidence < mLogZ + log10(mConfig.precisionCriterion)) {
        return true;
    }

    if (mIter >= mConfig.maxIters) {
        return true;
    }

    return false;
}




// returns log(A + B)
const double NestedSampler::logAdd(const double logA, const double logB) {
    if (logA > logB) {
        return logA + log(1 + exp(logB - logA));
    } else {
        return logB + log(1 + exp(logA - logB));
    }
}

// logAdd for vectors
const double NestedSampler::logAdd(const Eigen::ArrayXd &logV) {
    const double maxLogV = logV.maxCoeff();

    return maxLogV + log(1 + (logV - maxLogV).exp().sum());
}




