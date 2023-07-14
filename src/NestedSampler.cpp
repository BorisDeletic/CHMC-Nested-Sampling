#include "NestedSampler.h"
#include "Logger.h"
#include <iostream>
#include <cfloat>
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
        mUniformRng(0, 1),
        mLogImportanceWeight(log(exp(1.0L / mConfig.numLive) - 1.0L))
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

    mLogZ = -DBL_MAX; // Z = 0 initially
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
        mLogger.WritePoint(point, mLogImportanceWeight);
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
    mLogger.WritePoint(deadPoint, mLogImportanceWeight);

    // Generate new point(s)
    if ((double)deadPoint.reflections / deadPoint.steps > reflectionRateThreshold)
    {
        const MCPoint& randPoint = GetRandomLivePoint();
        SampleNewPoint(randPoint, deadPoint.likelihood);
    }
    else
    {
        SampleNewPoint(deadPoint, deadPoint.likelihood);
    }

    if (mConfig.logDiagnostics) {
        assert(mAdapter != nullptr);
        NSInfo info = {1,1,1.0,1.0,1.0};
        mLogger.WriteDiagnostics(info, deadPoint, *mAdapter);
    }

    //kill point.
    mLivePoints.erase(lowestIt);
}


void NestedSampler::SampleNewPoint(const MCPoint& deadPoint, const double likelihoodConstraint)
{
    for (int i = 0; i < mSampleRetries; i++) {
        const MCPoint newPoint = mSampler.SamplePoint(deadPoint, likelihoodConstraint);

        if (mAdapter != nullptr)
        {
            mAdapter->AdaptEpsilon(newPoint.acceptProbability);
        }

        if (!newPoint.rejected)
        {
            // sampled valid new point
            mLivePoints.insert(newPoint);

            if (mLivePoints.size() > mConfig.numLive)
                return;
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


void NestedSampler::UpdateLogEvidence(const MCPoint& point) {
    mLogImportanceWeight -= 1.0f / mLivePoints.size(); // compress space

    double logEvidence = mLogImportanceWeight + point.likelihood;

    mLogZ = logAdd(mLogZ, logEvidence);
}


// Estimate Log evidence remaining in live points
const double NestedSampler::EstimateLogEvidenceRemaining() {
    Eigen::ArrayXd logLikelihoodLive(mConfig.numLive);

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

    const NSInfo info = {
            mIter,
            mConfig.numLive,
            mConfig.reflectionRateThreshold,
            mLogZ,
            EstimateLogEvidenceRemaining()
    };

    return info;
}


const bool NestedSampler::TerminateSampling() {
    if (mAdapter != nullptr)
    {
        std::cout << "NS Step: " << mIter << ", Num Live = " << mLivePoints.size() << std::endl;
        std::cout << "e=" << mAdapter->GetEpsilon() << ", reflectionrate=" << GetReflectRate() << std::endl;

        mAdapter->AdaptMetric(mLivePoints);
    }

    double remainingEvidence = EstimateLogEvidenceRemaining();
    std::cout << "Step: " << mIter << std::endl;
    std::cout << "Log(Z)=" << mLogZ << " ,LogZlive=" << remainingEvidence << std::endl << std::endl;

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




