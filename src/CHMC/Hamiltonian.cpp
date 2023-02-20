#include "Hamiltonian.h"
#include "Likelihood.h"

Hamiltonian::Hamiltonian(const double epsilon)
    :
        mIntegrator(epsilon)
{
}

void Hamiltonian::SetHamiltonian(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const double likelihoodConstraint) {
  //  int size = x.size();

    mIntegrator.SetX(x);
    mIntegrator.SetP(p);

   // mGradient.resize(x.size());
    mLikelihoodConstraint = likelihoodConstraint;
}


void Hamiltonian::Evolve(const int steps) {

}

void Hamiltonian::UpdateGradient(const Eigen::VectorXd &x) {
//    const int size = x.size();



}


