#include <iostream>
#include "Hamiltonian.h"


Hamiltonian::Hamiltonian(ILikelihood& likelihood, IParams& params)
    :
        mLikelihood(likelihood),
        mParams(params),
        mIntegrator(mParams),
        mDimension(likelihood.GetDimension())
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
    mGradient = mLikelihood.Gradient(x);
}


void Hamiltonian::Evolve()
{
    if (mRejected) return;
    mIters++;

    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mGradient);
    const double newLikelihood = mLikelihood.LogLikelihood(newX);

    if (newLikelihood <= mLikelihoodConstraint) {
        // Reflect off iso-likelihood contour.
        mReflections++;

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
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

  //  Eigen::VectorXd nRot = invMetric.asDiagonal() * normal;
    Eigen::VectorXd nHat = normal.normalized();

  //  Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nRot) / normal.dot(nRot) * normal;

    double pdotn = mP.dot(nHat);
    double nMag = normal.norm();
    double pMag = mP.norm();

    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nHat) * nHat;

    mIntegrator.ChangeP(mP, reflectedP);
    mP = reflectedP;
}


void Hamiltonian::ReflectX(const Eigen::VectorXd &normal) {

    for (int i = 0; i < mEpsilonReflectionLimit; i++) {
        const double epsilonFactor = 1.0 / pow(2, i);
       // const double epsilonFactor = 1.0;

        Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mGradient, epsilonFactor);
        const double nextLikelihood = mLikelihood.LogLikelihood(nextX);

        if (nextLikelihood > mLikelihoodConstraint) {
            // found valid reflection

            mX = nextX;
            mLogLikelihood = nextLikelihood;

            return;
        }
      //  ReflectP(normal);
    }
    std::cout<<"NOREFLECTIONS" << std::endl;
    mRejected = true;

  //  throw std::runtime_error("NO VALID REFLECTION");
}


const double Hamiltonian::GetEnergy() const {
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

    const double energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP) - mLogLikelihood;
    return energy;
}



