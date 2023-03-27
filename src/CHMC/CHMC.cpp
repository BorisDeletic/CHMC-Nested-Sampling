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




Eigen::VectorXd CHMC::SampleP(const int size) {
    Eigen::VectorXd p(size);

    Eigen::VectorXd sqrtMetric = mParams.GetMetric().cwiseSqrt();

    for (int i = 0; i < size; i++) {
        p(i) = mNorm(gen) * sqrtMetric(i);
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

    const double acceptProb = exp(initEnergy - newEnergy);
    const double r = mUniform(gen);
    if ((acceptProb > r) && (!mHamiltonian.GetRejected()))
    {
        MCPoint newPoint = {
                mHamiltonian.GetX(),
                mHamiltonian.GetLikelihood(),
                likelihoodConstraint
        };
        return newPoint;
    }
    else
    {
        if (mHamiltonian.GetRejected()) {
            mReflectRejections++;
            std::cout << " !REFLECT! ";
        } else {
            mEnergyRejections++;
            std::cout << " !ENERGY! ";
        }
        MCPoint rejectedPoint = {
                Eigen::VectorXd::Zero(mLikelihood.GetDimension()),
                0,
                0,
                true
        };
        return rejectedPoint;
    }
}

const Rejections CHMC::GetRejections() {
    Rejections rejected = {
            mReflectRejections,
            mEnergyRejections,
            mIters
    };

    return rejected;
}


