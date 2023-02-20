#include "Hamiltonian.h"
#include "Likelihood.h"

Hamiltonian::Hamiltonian(const double epsilon, const int dimension)
    :
        mIntegrator(epsilon),
        mDimension(dimension)
{
    mGradient.resize(mDimension);
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
    assert((x.size() == mDimension) && (p.size() == mDimension));
    mIntegrator.SetX(x);
    mIntegrator.SetP(p);

    mLikelihoodConstraint = likelihoodConstraint;

    UpdateGradient(x);
}


void Hamiltonian::Evolve(const int steps) {

    for (int i = 0; i < steps; i++) {
        mIntegrator.UpdateX(mGradient);

        UpdateLikelihood(mIntegrator.GetX());
        UpdateGradient(mIntegrator.GetX());

        // Reflections off likehood countour here.

        mIntegrator.UpdateP(mGradient);
    }
}


void Hamiltonian::UpdateGradient(const Eigen::VectorXd &x) {
    memset(mGradient.data(), 0, mDimension * sizeof(double));

    Likelihood::gradient(const_cast<double *>(x.data()), mGradient.data(), mDimension);
}


void Hamiltonian::UpdateLikelihood(const Eigen::VectorXd &x) {
    mLikelihood = Likelihood::likelihood(const_cast<double *>(x.data()), mDimension);
}


