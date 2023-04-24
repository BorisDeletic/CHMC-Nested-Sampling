#include "NestedSampler.h"
#include "Logger.h"
#include <iostream>
#include <cfloat>


NestedSampler::NestedSampler(ISampler& sampler, ILikelihood& likelihood, Logger& logger,
                             NSConfig config)
    :
    mSampler(sampler),
    mLikelihood(likelihood),
    mLogger(logger),
    mConfig(config),
    mDimension(mLikelihood.GetDimension()),
    gen(rd()),
    mUniform(0,1),
    mLogWeight(log(exp(1.0L/mConfig.numLive) - 1.0L))
{
    mReflections = 0;
    mIntegrationSteps=0;
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
        mLogger.WritePoint(point, mLogWeight);
    }

    NSSummary summary = {
            mLogZ,
            EstimateLogEvidenceRemaining()
    };

    mLogger.WriteSummary(summary);

    int totalParams = mDimension + mLikelihood.DerivedParams(Eigen::VectorXd::Ones(mDimension)).size();
    mLogger.WriteParamNames(mLikelihood.ParamNames(), totalParams);
}


void NestedSampler::NestedSamplingStep() {
    auto lowestIt = mLivePoints.begin();
    const MCPoint &deadPoint = *lowestIt; // lowest likelihood point

    // Analysis and log dead point
    UpdateLogEvidence(deadPoint);
    mLogger.WritePoint(deadPoint, mLogWeight);

    // Generate new point(s)
    if ((double)deadPoint.reflections / deadPoint.steps > 0.9)
    {
        const MCPoint& randPoint = GetRandomPoint();
        SampleNewPoint(randPoint, deadPoint.likelihood);
    }
    else
    {
        SampleNewPoint(deadPoint, deadPoint.likelihood);
    }

    //kill point.
    mLivePoints.erase(lowestIt);

    if ((mAdapter != nullptr) && (mIter % 50 == 0)) {
        //mAdapter->AdaptEpsilon((double)mReflections / mIntegrationSteps);
        const double reflectionRate = (double) mReflections / mIntegrationSteps * 100;
    //    std::cout << "e=" << mAdapter->GetEpsilon() << ", reflectionrate=" << reflectionRate << ", iter=" << mIter
    //              << std::endl;

        mReflections = 0;
        mIntegrationSteps = 0;
    }
}


void NestedSampler::SampleNewPoint(const MCPoint& deadPoint, const double likelihoodConstraint)
{
    for (int i = 0; i < mSampleRetries; i++) {
        const MCPoint newPoint = mSampler.SamplePoint(deadPoint, likelihoodConstraint);

        if (mAdapter != nullptr)
        {
            mAdapter->AdaptEpsilon(newPoint.acceptProbability);
            mReflections += newPoint.reflections;
            mIntegrationSteps += newPoint.steps;
        }

        if (!newPoint.rejected)
        {
            // sampled valid new point
            mLivePoints.insert(newPoint);

            if (mLivePoints.size() > mConfig.numLive)
            {
                return;
            }
        }
    }
}


const MCPoint NestedSampler::SampleFromPrior() {
    Eigen::VectorXd cube = Eigen::VectorXd::NullaryExpr(mDimension, [&](){
        return mUniform(gen);
    });
    Eigen::VectorXd theta = mLikelihood.PriorTransform(cube);

    MCPoint pointFromPrior = {
            theta,
            mLikelihood.DerivedParams(theta),
            mLikelihood.LogLikelihood(theta),
            minLikelihood
    };

    return pointFromPrior;
}


const MCPoint& NestedSampler::GetRandomPoint() {
    const int randomIndex = std::floor(mUniform(gen) * mLivePoints.size());

    auto It = mLivePoints.begin();
    std::advance(It, randomIndex);

    return *It;
}



void NestedSampler::UpdateLogEvidence(const MCPoint& point) {
    mLogWeight -= 1.0f / mLivePoints.size(); // compress space

    double logEvidence = mLogWeight + point.likelihood;

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

    double logEvidenceLive = logMeanLiveLikelihood - mIter / mConfig.numLive;
    return logEvidenceLive;
}


const bool NestedSampler::TerminateSampling() {
    if (mAdapter != nullptr)
    {
        const double reflectionRate = (double) mReflections / mIntegrationSteps * 100;
        std::cout << "NS Step: " << mIter << ", Num Live = " << mLivePoints.size() << std::endl;
        std::cout << "e=" << mAdapter->GetEpsilon() << ", reflectionrate=" << reflectionRate << std::endl;

        mReflections = 0;
        mIntegrationSteps = 0;

        mAdapter->AdaptMetric(mLivePoints);
      //  mAdapter->Restart();
    }

    double remainingEvidence = EstimateLogEvidenceRemaining();
    std::cout << "Log(Z)=" << mLogZ << " ,LogZlive=" << remainingEvidence << std::endl;

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



