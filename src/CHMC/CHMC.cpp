#include "CHMC.h"
#include "Hamiltonian.h"
#include <memory>
#include <iostream>

CHMC::CHMC(ILikelihood& likelihood, double epsilon, int pathLength)
    :
    mLikelihood(likelihood),
    mPathLength(pathLength),
    gen(rd()),
    mNorm(0,1),
    mUniform(0, 1)
{
    mHamiltonian = std::make_unique<Hamiltonian>(likelihood, epsilon);
}

CHMC::~CHMC() = default;


Eigen::VectorXd CHMC::SampleMomentum(const int size) {
    Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(size, [&](){
        return mNorm(gen);
    });

    return v;
}


const MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    const Eigen::VectorXd p = SampleMomentum(old.theta.size());

    mHamiltonian->SetHamiltonian(old.theta, p, likelihoodConstraint);
    const double initEnergy = mHamiltonian->GetEnergy();

    for (int i = 0; i < mPathLength; i++) {
        mHamiltonian->Evolve();
    }

    const double acceptProb = exp(initEnergy - mHamiltonian->GetEnergy());
    const double r = mUniform(gen);
    if (acceptProb > r)
    {
        MCPoint newPoint = {
                mHamiltonian->GetX(),
                mHamiltonian->GetLikelihood(),
                likelihoodConstraint
        };
        return newPoint;

    } else
    {
        std::cerr << "REJECT POINT";
        return old;
    }
}

