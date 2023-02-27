#include "NestedSampler.h"
#include "Logger.h"
#include <stdexcept>


NestedSampler::NestedSampler(CHMC& chmc, IPrior& prior, ILikelihood& likelihood,
                             int numLive, std::string name)
    :
    mCHMC(chmc),
    mPrior(prior),
    mLikelihood(likelihood),
    mNumLive(numLive),
    mDimension(mCHMC.GetDimension()),
    gen(rd()),
    mUniform(0,1),
    mName(name)
{
    mLogger = std::make_unique<Logger>(mName);
    if (mCHMC.GetDimension() != mPrior.GetDimension()) {
        throw std::runtime_error("LIKELIHOOD AND PRIOR HAVE DIFFERENT DIMENSIONS");
    }
}

void NestedSampler::Initialise() {
    for (int i = mLivePoints.size(); i < mNumLive; i++) {
        MCPoint newPoint = SampleFromPrior();
        mLivePoints.insert(newPoint);
    }
}

void NestedSampler::Run(int steps) {

}

MCPoint NestedSampler::SampleFromPrior() {
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


