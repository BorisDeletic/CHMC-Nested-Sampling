#include "Hamiltonian.h"


Hamiltonian::Hamiltonian(ILikelihood& likelihood, const double epsilon)
    :
        mLikelihood(likelihood),
        mIntegrator(epsilon)
{
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
    mIntegrator.SetX(x);
    mIntegrator.SetP(p);

    mLikelihoodConstraint = likelihoodConstraint;

    mGradient = mLikelihood.Gradient(mIntegrator.GetX());
}


void Hamiltonian::Evolve(const int steps)
{
    for (int i = 0; i < steps; i++)
    {
        mIntegrator.UpdateX(mGradient);

        mCurrentLikelihood = mLikelihood.Likelihood(mIntegrator.GetX());
        mGradient = mLikelihood.Gradient(mIntegrator.GetX());

        // Reflections off likelihood contour here.

        mIntegrator.UpdateP(mGradient);
    }
}


