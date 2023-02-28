#include "RejectionSampler.h"

RejectionSampler::RejectionSampler(ILikelihood& likelihood, double epsilon)
    :
    mLikelihood(likelihood),
    mEpsilon(epsilon),
    gen(rd()),
    mUniform(0,1)
{}

const MCPoint RejectionSampler::SamplePoint(const MCPoint &old, const double likelihoodConstraint)
{
    double stepSize = mEpsilon;
    int chainSteps = 20;
    int accepts = 0;
    int rejections = 0;

    // seed new point with old point
    Eigen::VectorXd newTheta = old.theta;
    double newLikelihood = old.likelihood;


    for (int i = 0; i < chainSteps; i++) {
        Eigen::VectorXd rand = Eigen::VectorXd::NullaryExpr(old.theta.size(), [&](){
            return 2 * mUniform(gen) - 1;
        });

        Eigen::VectorXd trialTheta = newTheta + stepSize * rand;
        const double trialLikelihood = mLikelihood.Likelihood(trialTheta);

        // Only accept if L > Lconstraint
        if (trialLikelihood > likelihoodConstraint) {
            accepts++;
            newTheta = trialTheta;
            newLikelihood = trialLikelihood;
        } else {
            rejections++;
        }

        // Modify step size to target 50% acceptance rate
        if ( accepts > rejections ) {
            stepSize *= exp(1.0 / accepts);
        } else {
            stepSize /= exp(1.0 / rejections);
        }
    }

    MCPoint newPoint {
        newTheta,
        newLikelihood,
        likelihoodConstraint
    };

    return newPoint;
}