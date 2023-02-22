#include "CHMC.h"

#include <memory>
#include "Hamiltonian.h"

CHMC::CHMC(ILikelihood& likelihood, double epsilon, int pathLength)
    :
    mPathLength(pathLength),
    gen(rd()),
    mNorm(0,1)
{
    mHamiltonian = std::make_unique<Hamiltonian>(likelihood, epsilon);
}

// Normally distributed
Eigen::VectorXd CHMC::SampleMomentum(const int size) {
    Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(size, [&](){
        return mNorm(gen);
    });

    return v;
}


MCPoint CHMC::SamplePoint(const MCPoint &old, double likelihoodConstraint) {
    Eigen::Map<const Eigen::VectorXd> x(old.theta, old.size);
    const Eigen::VectorXd p = SampleMomentum(old.size);

    mHamiltonian->SetHamiltonian(x, p, likelihoodConstraint);
    mHamiltonian->Evolve(mPathLength);

    const double* newTheta = static_cast<const double *>(mHamiltonian->GetX().data());

    MCPoint newPoint = {
            newTheta,
            old.size,
            mHamiltonian->GetLikelihood()
    };

    return newPoint;
}
