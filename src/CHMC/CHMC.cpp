#include "CHMC.h"
#include "Hamiltonian.h"
#include <iostream>

CHMC::CHMC(IPrior& prior, ILikelihood& likelihood, IParams& params)
    :
    mParams(params),
    mLikelihood(likelihood),
    mHamiltonian(prior, likelihood, params),
    gen(rd()),
    mNorm(0, 1),
    mUniform(0, 1)
{
}



// P ~ N(0, sigma=M)
Eigen::VectorXd CHMC::SampleP(const int size) {
    Eigen::VectorXd p(size);

    Eigen::VectorXd metric = mParams.GetMetric();

    for (int i = 0; i < size; i++) {
        p(i) = mNorm(gen) * metric(i);
    }

    return p;
}


const MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    mIters++;

    const Eigen::VectorXd p = SampleP(old.theta.size());

    mHamiltonian.SetHamiltonian(old.theta, p, likelihoodConstraint);
    const double initEnergy = mHamiltonian.GetEnergy();

    for (int i = 0; i < mParams.GetPathLength(); i++) {
        mHamiltonian.Evolve();
    }

    const double newEnergy = mHamiltonian.GetEnergy();
    double acceptProb = exp(initEnergy - newEnergy);
    const double r = mUniform(gen);
    bool rejected = mHamiltonian.GetRejected();

    if (rejected) {
        rejected = true;
        acceptProb = 0;
    }

    if (acceptProb < r) {
        rejected = true;
    }

    const double reflectionRate = (double)mHamiltonian.GetReflections() / mHamiltonian.GetIntegrationSteps();
    if (reflectionRate > 0.9) {
        acceptProb = 0;
    }

    MCPoint newPoint = {
            mHamiltonian.GetX(),
            mLikelihood.DerivedParams(mHamiltonian.GetX()),
            mHamiltonian.GetLikelihood(),
            likelihoodConstraint,
            mHamiltonian.GetReflections(),
            mHamiltonian.GetIntegrationSteps(),
            acceptProb,
            rejected
    };

    return newPoint;
}



