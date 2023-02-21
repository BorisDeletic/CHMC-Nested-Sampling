#include "CHMC.h"

CHMC::CHMC(ILikelihood& likelihood, double epsilon, int pathLength)
    :
    mHamiltonian(likelihood, epsilon),
    mPathLength(pathLength),
    gen(rd()),
    mNorm(0,1)
{
}

// Normally distributed
Eigen::VectorXd CHMC::SampleMomentum(const int size) {
    Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(size, [&](){
        return mNorm(gen);
    });

    return v;
}

MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    Eigen::VectorXd p = SampleMomentum(old.size);
    Eigen::VectorXd x =

    return MCPoint();
}
