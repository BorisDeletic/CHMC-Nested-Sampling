#include "Hamiltonian.h"


Hamiltonian::Hamiltonian(ILikelihood& likelihood, const double epsilon)
    :
        mLikelihood(likelihood),
        mIntegrator(epsilon)
{
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
    mX = x;
    mP = p;

    mLikelihoodConstraint = likelihoodConstraint;

    mLogLikelihood = mLikelihood.LogLikelihood(x);
    mGradient = mLikelihood.Gradient(x);

    mEnergy = 0.5 * mP.squaredNorm() - mLogLikelihood;
}


void Hamiltonian::Evolve()
{
    Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mGradient);

    const double newLikelihood = mLikelihood.LogLikelihood(newX);
    if (newLikelihood < mLikelihoodConstraint) {
        // Reflect off iso-likelihood contour.

        Eigen::VectorXd newP = ReflectP(mP, mGradient);
        mIntegrator.ChangeP(mP, newP);
        mP = newP;

        Eigen::VectorXd nextX = mIntegrator.UpdateX(mX, mP, mGradient);
        if (mLikelihood.LogLikelihood(nextX) < mLikelihoodConstraint) {
           // throw std::runtime_error("NO VALID REFLECTION");
        }
    }
    else {
        mX = newX;
        mLogLikelihood = newLikelihood;
    }

    mGradient = mLikelihood.Gradient(mX);

    mP = mIntegrator.UpdateP(mX, mP, mGradient);
    mEnergy = 0.5 * mP.squaredNorm() - mLogLikelihood;

}


//incident momentum and normal vector to reflection boundary
Eigen::VectorXd
Hamiltonian::ReflectP(const Eigen::VectorXd &incidentP, const Eigen::VectorXd &normal) {
    Eigen::VectorXd nHat = normal.normalized();

    Eigen::VectorXd reflectedP = incidentP - 2 * incidentP.dot(nHat) * nHat;

    return reflectedP;
}



