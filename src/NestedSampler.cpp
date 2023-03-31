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
    initialLogWeight(log(exp(1/mConfig.numLive - 1)))
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
        mLogger.WritePoint(newPoint);
    }

    mLogZ = -DBL_MAX; // Z = 0 initially
}


void NestedSampler::SetAdaption(Adapter* adapter) {
    mAdapter = adapter;
}



void NestedSampler::Run() {
    bool terminationCondition = false;
    while (!terminationCondition) {
     //   std::cout << ", Reject Ratio = " << mSampler.GetSummary().rejectRatio << std::endl;

        NestedSamplingStep();
        mIter++;

        if (mIter % mConfig.numLive == 0) {
            terminationCondition = TerminateSampling();
        }

        if (mIter % 50 == 0) {
            std::cout << "NS Step: " << mIter;
            std::cout << ", Num Live = " << mLivePoints.size() << std::endl;
        }
    }

    NSSummary summary = {
            mLogZ,
            EstimateLogEvidenceRemaining()
    };

  //  SamplerSummary samplerStats = mSampler.GetSummary();

    mLogger.WriteSummary(summary);
}


void NestedSampler::NestedSamplingStep() {
    auto lowestIt = mLivePoints.begin();
    const MCPoint& deadPoint = *lowestIt;

    // Analysis on dead point
    UpdateLogEvidence(deadPoint.likelihood);

    // Generate new point(s)
    SampleNewPoint(deadPoint);

    //kill point.
    mLivePoints.erase(lowestIt);

    if ((mAdapter != nullptr) && (mIter % 50 == 0)) {
    //    mAdapter->AdaptEpsilon((double)mReflections / mIntegrationSteps);
     //   const double reflectionRate = (double)mReflections / mIntegrationSteps;
     //   std::cout << "e=" << mAdapter->GetEpsilon() << ", reflectionrate=" << reflectionRate << ", iter=" << mIter << std::endl;

        mReflections = 0;
        mIntegrationSteps=0;
        std::cout << "NS Step: " << mIter;
        std::cout << ", Num Live = " << mLivePoints.size() << std::endl;

    }
}


void NestedSampler::SampleNewPoint(const MCPoint& deadPoint) {
    const double likelihoodConstraint = deadPoint.likelihood;

    for (int i = 0; i < mSampleRetries; i++) {
        const MCPoint newPoint = mSampler.SamplePoint(deadPoint, likelihoodConstraint);

        if (mAdapter != nullptr)
        {
       //     mAdapter->AdaptEpsilon(newPoint.acceptProbability);
            mReflections += newPoint.reflections;
            mIntegrationSteps += newPoint.steps;
        }

        if (!newPoint.rejected)
        {
            // sampled valid new point
            mLivePoints.insert(newPoint);
            mLogger.WritePoint(newPoint);

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
            mLikelihood.LogLikelihood(theta),
            minLikelihood
    };

    return pointFromPrior;
}


void NestedSampler::UpdateLogEvidence(const double logLikelihood) {
    double logWeight = initialLogWeight - (float)mIter / mConfig.numLive;

    double logEvidence = logWeight + logLikelihood;

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

    double logLikelihoodRemaining = logAdd(logLikelihoodLive);
    double logWeight = initialLogWeight - (float)mIter / mConfig.numLive;

    double logEvidenceLive = logLikelihoodRemaining + logWeight;
    return logEvidenceLive;
}


const bool NestedSampler::TerminateSampling() {
    double remainingEvidence = EstimateLogEvidenceRemaining();
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



