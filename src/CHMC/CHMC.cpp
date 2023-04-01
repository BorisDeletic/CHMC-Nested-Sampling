#include "CHMC.h"
#include "Hamiltonian.h"
#include <memory>
#include <cfloat>
#include <iostream>

CHMC::CHMC(ILikelihood& likelihood, IParams& params)
    :
    mLikelihood(likelihood),
    mParams(params),
    mHamiltonian(likelihood, params),
    gen(rd()),
    mNorm(0, 1),
    mUniform(0, 1)
{
}



// P ~ N(0, M)
Eigen::VectorXd CHMC::SampleP(const int size) {
    Eigen::VectorXd p(size);

    Eigen::VectorXd metric = mParams.GetMetric();

    for (int i = 0; i < size; i++) {
        std::normal_distribution<double> normalDistribution(0, metric(i));

       // p(i) = mNorm(gen) * metric(i);
        p(i) = normalDistribution(gen);
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

   // std::cout << "e=" << mParams.GetEpsilon() << ", prob=" << acceptProb << ", ";
    const double reflectionRate = (double)mHamiltonian.GetReflections() / mHamiltonian.GetIntegrationSteps();
    if (rejected) {
        rejected = true;
        acceptProb = 0;
        std::cout << "!reflect=" << reflectionRate << std::endl;
    }


    if (acceptProb < r) {
        rejected = true;
    }


    MCPoint newPoint = {
            mHamiltonian.GetX(),
            mHamiltonian.GetLikelihood(),
            likelihoodConstraint,
            mHamiltonian.GetReflections(),
            mHamiltonian.GetIntegrationSteps(),
            acceptProb,
            rejected,
            newEnergy
    };

    return newPoint;
}

const Rejections CHMC::GetRejections() {
    Rejections rejected = {
            mReflectRejections,
            mEnergyRejections,
            mIters
    };

    return rejected;
}


