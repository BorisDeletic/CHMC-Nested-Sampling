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

   mLikelihoodGradient = mLikelihood.Gradient(x);
   if (mUsePosterior) {
       mPriorGradient = mLikelihood.Gradient(x);
   } else {
       mPriorGradient = mPrior.Gradient(x);
   }

    dxs.clear();
    likes.clear();
    energies.clear();
}


void Hamiltonian::Evolve()
{
    if (mRejected) return;
    mIters++;

    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mPriorGradient);
    const double newLikelihood = mLikelihood.LogLikelihood(newX);

    if ((newLikelihood <= mLikelihoodConstraint) || OutsidePriorBounds(newX)) {
        // Reflect off iso-likelihood contour OR prior boundary.
        Reflection();

       // mLikelihoodGradient = mLikelihood.Gradient(mX);

 //       ReflectP(mLikelihoodGradient);
   //     ReflectX(mLikelihoodGradient);
    }
    else
    {
        double dx = (mX - newX).norm();
        dxs.push_back(dx);
        likes.push_back(newLikelihood);
        energies.push_back(GetEnergy());

        mX = newX;
        mLogLikelihood = newLikelihood;
    }

    if (mUsePosterior) {
        mPriorGradient = mLikelihood.Gradient(mX);
    } else {
        mPriorGradient = mPrior.Gradient(mX);
    }

    mP = mIntegrator.UpdateP( mPriorGradient);
}


//incident momentum and normal vector to reflection boundary
void Hamiltonian::ReflectP(const Eigen::VectorXd &normal) {

    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

 //   Eigen::VectorXd nRot = invMetric.asDiagonal() * normal;

//    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(nRot) / normal.dot(nRot) * normal;
    Eigen::VectorXd reflectedP = mP - 2 * mP.dot(normal) / normal.dot(normal) * normal;

    mIntegrator.ChangeP(mP, reflectedP);
    mP = reflectedP;
}


void Hamiltonian::Reflection() {
    mReflections++;

    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mPriorGradient);
    const double newLikelihood = mLikelihood.LogLikelihood(newX);

    if (newLikelihood <= mLikelihoodConstraint) {
        ReflectP(mLikelihood.Gradient(mX));
    }

    if (OutsidePriorBounds(newX)) {
        Eigen::VectorXd priorNormal = GetPriorReflection(newX);
        ReflectP(priorNormal);
    }

    for (int i = 0; i < mEpsilonReflectionLimit; i++) {
        const double epsilonFactor = 1.0 / pow(2, i);

        Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mPriorGradient, epsilonFactor);
        const double newLikelihood = mLikelihood.LogLikelihood(newX);

        if ((newLikelihood > mLikelihoodConstraint) && (!OutsidePriorBounds(newX))) {
//        if (nextLikelihood > mLikelihoodConstraint) {
            // found valid reflection
            double dx = (mX - newX).norm();
            dxs.push_back(dx);
            likes.push_back(newLikelihood);
            energies.push_back(GetEnergy());

            mX = newX;
            mLogLikelihood = newLikelihood;

            return;
        }

        ReflectP(mPriorGradient);
    }

    Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mPriorGradient);
    const double nextLikelihood = mLikelihood.LogLikelihood(nextX);

    double dx = (mX - nextX).norm();
    dxs.push_back(dx);
    likes.push_back(nextLikelihood);
    energies.push_back(GetEnergy());

//    std::cout << std::endl << "|n| = " << mLikelihood.Gradient(mX).norm() << std::endl;
//    std::cout << "|p| = ";
//    std::cout << mP.norm() << std::endl;
//    std::cout << "Mx = " << (mX).norm() << std::endl;
//    std::cout << "Nx = " << (nextX).norm() << std::endl;
//    std::cout << "P = " << mP.norm() << std::endl;
//    std::cout << "ef = " << 1.0/pow(2,10) << std::endl;

    mRejected = true;
}


double Hamiltonian::GetEnergy() const {
    const Eigen::VectorXd invMetric = mParams.GetMetric().cwiseInverse();

//  need to make this function work with pi(theta) not being uniform
    double energy;

    if (mUsePosterior) {
        energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP) - mLogLikelihood;
    } else {
        energy = 0.5 * mP.dot(invMetric.asDiagonal() * mP);
    }

    return energy;
}


bool Hamiltonian::OutsidePriorBounds(const Eigen::VectorXd &theta) {
    // this is a hack which assumes the prior function is min@0 and max@1

    if (mUsePosterior) {
        return false;
    }

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



