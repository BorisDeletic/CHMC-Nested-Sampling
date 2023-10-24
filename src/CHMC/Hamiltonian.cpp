#include <iostream>
#include "Hamiltonian.h"


Hamiltonian::Hamiltonian(IPrior& prior, ILikelihood& likelihood, IParams& params)
    :
        mPrior(prior),
        mLikelihood(likelihood),
        mParams(params),
        mIntegrator(mParams)
{
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
    mX = x;
    mP = p;
    mRejected = false;
    mReflections = 0;
    mIters = 0;

    mLikelihoodConstraint = likelihoodConstraint;
    mLogLikelihood = mLikelihood.LogLikelihood(x);

   // mLikelihoodGradient = mLikelihood.Gradient(x);
    mPriorGradient = mPrior.Gradient(x);
}


void Hamiltonian::Evolve()
{
    if (mRejected) return;
    mIters++;

    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mPriorGradient);
    const double newLikelihood = mLikelihood.LogLikelihood(newX);

    if (newLikelihood <= mLikelihoodConstraint) {
        // Reflect off iso-likelihood contour.

        mLikelihoodGradient = mLikelihood.Gradient(mX);

        ReflectP(mLikelihoodGradient);
        ReflectX(mLikelihoodGradient);
    }
    if (OutsidePriorBounds(newX)) {
        Eigen::VectorXd reflect = GetPriorReflection(newX);
        ReflectP(reflect);
        ReflectX(reflect);
    }
    else
    {
        mX = newX;
        mLogLikelihood = newLikelihood;
    }

    mPriorGradient = mPrior.Gradient(mX);

    mP = mIntegrator.UpdateP( mPriorGradient);
}


//incident momentum and normal vector to reflection boundary
void Hamiltonian::ReflectP(const Eigen::VectorXd &normal) {
    mReflections++;

    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

 //   Eigen::VectorXd nRot = invMetric.asDiagonal() * normal;

//    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nRot) / normal.dot(nRot) * normal;
    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(normal) / normal.dot(normal) * normal;

    mIntegrator.ChangeP(mP, reflectedP);
    mP = reflectedP;
}


void Hamiltonian::ReflectX(const Eigen::VectorXd &normal) {

    for (int i = 0; i < mEpsilonReflectionLimit; i++) {
        const double epsilonFactor = 1.0 / pow(2, i);

        Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mPriorGradient, epsilonFactor);
        const double nextLikelihood = mLikelihood.LogLikelihood(nextX);

        if ((nextLikelihood > mLikelihoodConstraint) && (!OutsidePriorBounds(nextX))) {
            // found valid reflection

            mX = nextX;
            mLogLikelihood = nextLikelihood;

            return;
        }
        ReflectP(normal);
    }

    std::cout  << normal.norm() << std::endl;
 //   std::cout << "x = ";
 //   std::cout << mX << std::endl;
    mRejected = true;
}


const double Hamiltonian::GetEnergy() const {
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

//  need to make this function work with pi(theta) not being uniform

//    const double energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP) - mLogLikelihood;
    const double energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP);

    return energy;
}


bool Hamiltonian::OutsidePriorBounds(const Eigen::VectorXd &theta) {
    // this is a hack which assumes the prior function is min@0 and max@1

    Eigen::ArrayXd lowerBound = mPrior.PriorTransform(Eigen::VectorXd::Zero(mPrior.GetDimension()));
    Eigen::ArrayXd upperBound = mPrior.PriorTransform(Eigen::VectorXd::Ones(mPrior.GetDimension()));

    if ((theta.array() < lowerBound).any()) {
        return true;
    }
    if ((theta.array() > upperBound).any()) {
        return true;
    }

    return false;
}


const Eigen::VectorXd Hamiltonian::GetPriorReflection(const Eigen::VectorXd &theta) {
    Eigen::ArrayXd lowerBound = mPrior.PriorTransform(Eigen::VectorXd::Zero(mPrior.GetDimension()));
    Eigen::ArrayXd upperBound = mPrior.PriorTransform(Eigen::VectorXd::Ones(mPrior.GetDimension()));

    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(mPrior.GetDimension());

    for (int i = 0; i < theta.size(); i++) {
        if (theta[i] < lowerBound[i])
            gradient[i] = 1;

        if (theta[i] > upperBound[i])
            gradient[i] = -1;
    }

    return gradient;
}



