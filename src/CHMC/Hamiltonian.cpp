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
        mReflections++;

        mLikelihoodGradient = mLikelihood.Gradient(mX);

        ReflectP(mLikelihoodGradient);
        ReflectX(mLikelihoodGradient);
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
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

    Eigen::VectorXd nRot = invMetric.asDiagonal() * normal;

    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nRot) / normal.dot(nRot) * normal;
 //   Eigen::VectorXd reflectedP = mP - 2 * mP.dot(normal) / normal.dot(normal) * normal;

    mIntegrator.ChangeP(mP, reflectedP);
    mP = reflectedP;
}


void Hamiltonian::ReflectX(const Eigen::VectorXd &normal) {

    for (int i = 0; i < mEpsilonReflectionLimit; i++) {
        const double epsilonFactor = 1.0 / pow(2, i);

        Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mPriorGradient, epsilonFactor);
        const double nextLikelihood = mLikelihood.LogLikelihood(nextX);

        if (nextLikelihood > mLikelihoodConstraint) {
            // found valid reflection

            mX = nextX;
            mLogLikelihood = nextLikelihood;

            return;
        }
        ReflectP(normal);
    }

    mRejected = true;
}


const double Hamiltonian::GetEnergy() const {
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

//    const double energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP) - mLogLikelihood;
    const double energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP);

    return energy;
}



