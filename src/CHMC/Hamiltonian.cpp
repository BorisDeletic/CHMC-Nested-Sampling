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

    mForce = - mLikelihood.Gradient(x);
}


void Hamiltonian::Evolve(const int steps)
{
    for (int i = 0; i < steps; i++)
    {
        Eigen::VectorXd newX = mIntegrator.UpdateX(mX, mP, mForce);

        const double newLikelihood = mLikelihood.LogLikelihood(newX);
        if (newLikelihood < mLikelihoodConstraint) {
            // Reflect off iso-likelihood contour.

            double x0 = mX[0];
            double x1 = mX[1];
            double p0 = mP[0];
            double p1 = mP[1];
            double mag = mP.norm();
            Eigen::VectorXd newP = ReflectP(mP, mForce);
            mIntegrator.ChangeP(mP, newP);
            mP = newP;
        }
        else {
            mX = newX;
            mCurrentLikelihood = newLikelihood;
        }

        mForce = -mLikelihood.Gradient(mX);

        mP = mIntegrator.UpdateP(mX, mP, mForce);
    }
}


//incident momentum and normal vector to reflection boundary
Eigen::VectorXd
Hamiltonian::ReflectP(const Eigen::VectorXd &incidentP, const Eigen::VectorXd &normal) {
    Eigen::VectorXd nHat = normal.normalized();

    Eigen::VectorXd reflectedP = incidentP - 2 * incidentP.dot(nHat) * nHat;

    return reflectedP;
}



