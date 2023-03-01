#include "NestedSampler.h"
#include "Logger.h"
#include <stdexcept>
#include <iostream>
#include <cfloat>


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
    mUniform(0,1),
    initialLogWeight(log(exp(1/mConfig.numLive - 1)))
{
    if (mDimension != mPrior.GetDimension()) {
        throw std::runtime_error("LIKELIHOOD AND PRIOR HAVE DIFFERENT DIMENSIONS");
    }

}

NestedSampler::~NestedSampler() = default;


void NestedSampler::Initialise() {
    mIter = 0;
    mLivePoints.clear();

    for (int i = mLivePoints.size(); i < mConfig.numLive; i++) {
        MCPoint newPoint = SampleFromPrior();
        mLivePoints.insert(newPoint);
    }

    mLogZ = -DBL_MAX; // Z = 0 initially
}


void NestedSampler::Run() {
    bool terminationCondition = false;
    while (!terminationCondition) {
        std::cout << "NS Step: " << mIter << std::endl;
        NestedSamplingStep();
        mIter++;

        if (mIter % mConfig.numLive == 0) {
            terminationCondition = TerminateSampling();
        }
    }

    NSSummary summary = {
            mLogZ
    };

    mLogger.WriteSummary(summary);
}


void NestedSampler::NestedSamplingStep() {
    auto lowestIt = mLivePoints.begin();
    const MCPoint& deadPoint = *lowestIt;
    const double likelihoodConstraint = deadPoint.likelihood;

    // Analysis on dead point
    mLogger.WriteDeadPoint(deadPoint);
    UpdateLogEvidence(deadPoint.likelihood);

    const MCPoint newPoint = mSampler.SamplePoint(deadPoint, likelihoodConstraint);

    mLivePoints.erase(lowestIt);
    mLivePoints.insert(newPoint);
}


const MCPoint NestedSampler::SampleFromPrior() {
    Eigen::VectorXd cube = Eigen::VectorXd::NullaryExpr(mDimension, [&](){
        return mUniform(gen);
    });
    Eigen::VectorXd theta = mPrior.PriorTransform(cube);

    MCPoint pointFromPrior = {
            theta,
            mLikelihood.Likelihood(theta),
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
    if (EstimateLogEvidenceRemaining() < mLogZ + log(mConfig.precisionCriterion)) {
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

