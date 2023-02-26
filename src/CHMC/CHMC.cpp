#include "CHMC.h"
#include "Hamiltonian.h"
#include <memory>

CHMC::CHMC(ILikelihood& likelihood, double epsilon, int pathLength)
    :
    mPathLength(pathLength),
    gen(rd()),
    mNorm(0,1)
{
    mHamiltonian = std::make_unique<Hamiltonian>(likelihood, epsilon);
}

CHMC::~CHMC() = default;

// Normally distributed
Eigen::VectorXd CHMC::SampleMomentum(const int size) {
    Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(size, [&](){
        return mNorm(gen);
    });

    return v;
}


MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    const Eigen::VectorXd p = SampleMomentum(old.theta.size());

    mHamiltonian->SetHamiltonian(old.theta, p, likelihoodConstraint);
    mHamiltonian->Evolve(mPathLength);

    MCPoint newPoint = {
            mHamiltonian->GetX(),
            mHamiltonian->GetLikelihood()
    };

    return newPoint;
}

