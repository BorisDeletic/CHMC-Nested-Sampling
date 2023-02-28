#include "NestedSampler.h"
#include "Logger.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>


NestedSampler::NestedSampler(ISampler& sampler, IPrior& prior, ILikelihood& likelihood, Logger& logger,
                             int numLive)
    :
    mSampler(sampler),
    mPrior(prior),
    mLikelihood(likelihood),
    mLogger(logger),
    mNumLive(numLive),
    mDimension(mLikelihood.GetDimension()),
    gen(rd()),
    mUniform(0,1)
{
    if (mDimension != mPrior.GetDimension()) {
        throw std::runtime_error("LIKELIHOOD AND PRIOR HAVE DIFFERENT DIMENSIONS");
    }
}

NestedSampler::~NestedSampler() = default;


void NestedSampler::Initialise() {
    for (int i = mLivePoints.size(); i < mNumLive; i++) {
        MCPoint newPoint = SampleFromPrior();
        mLivePoints.insert(newPoint);
    }
}


void NestedSampler::Run(int steps) {
    for (int i = 0; i < steps; i++) {
        std::cout << "NS Step: " << i << std::endl;
        NestedSamplingStep();
    }
}


void NestedSampler::NestedSamplingStep() {
    auto lowestIt = mLivePoints.begin();
    const MCPoint& deadPoint = *lowestIt;
    const double likelihoodConstraint = deadPoint.likelihood;

    mLogger.WriteDeadPoint(deadPoint);

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
            mLikelihood.Likelihood(theta)
    };

    return pointFromPrior;
}
