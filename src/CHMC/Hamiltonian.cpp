#include <iostream>
#include "Hamiltonian.h"


Hamiltonian::Hamiltonian(ILikelihood& likelihood, const double epsilon)
    :
        mLikelihood(likelihood),
        mIntegrator(epsilon),
        mDimension(likelihood.GetDimension())
{
    mMetric = Eigen::VectorXd::Ones( mDimension);
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
    mX = x;
    mP = p;
    mRejected = false;

    mLikelihoodConstraint = likelihoodConstraint;

    mLogLikelihood = mLikelihood.LogLikelihood(x);
    mGradient = mLikelihood.Gradient(x);
}


void Hamiltonian::Evolve()
{
    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mGradient, mMetric);
    const double newLikelihood = mLikelihood.LogLikelihood(newX);

    if (newLikelihood <= mLikelihoodConstraint) {
        // Reflect off iso-likelihood contour.

        ReflectP(mGradient);
        ReflectX(mGradient);
    }
    else
    {
        mX = newX;
        mLogLikelihood = newLikelihood;
    }

    mGradient = mLikelihood.Gradient(mX);

    mP = mIntegrator.UpdateP( mGradient);
}


//incident momentum and normal vector to reflection boundary
void Hamiltonian::ReflectP(const Eigen::VectorXd &normal) {
    Eigen::VectorXd nRot = mMetric.cwiseInverse().asDiagonal() * normal;
 //   Eigen::VectorXd nHat = normal.normalized();

    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nRot) / normal.dot(nRot) * normal;
  //  Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nHat) * nHat;

    mIntegrator.ChangeP(mP, reflectedP);
    mP = reflectedP;
}


void Hamiltonian::ReflectX(const Eigen::VectorXd &normal) {
    const double oldEpsilon = mIntegrator.GetEpsilon();

    for (int i = 0; i < mEpsilonReflectionLimit; i++) {
        const double newEpsilon = oldEpsilon / (pow(2, i));
        mIntegrator.SetEpsilon(newEpsilon);

        Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mGradient, mMetric);
        const double nextLikelihood = mLikelihood.LogLikelihood(nextX);

        if (nextLikelihood > mLikelihoodConstraint) {
            // found valid reflection
            mFailedReflections += i;

            mX = nextX;
            mLogLikelihood = nextLikelihood;

            mIntegrator.SetEpsilon(oldEpsilon);
            return;
        }
        ReflectP(normal);
    }

    mIntegrator.SetEpsilon(oldEpsilon);
    mRejected = true;
  //  throw std::runtime_error("NO VALID REFLECTION");
}


const double Hamiltonian::GetEnergy() const {
    double energy = 0.5 * mP.dot(mMetric.cwiseInverse().asDiagonal() * mP) - mLogLikelihood;
    return energy;
}



