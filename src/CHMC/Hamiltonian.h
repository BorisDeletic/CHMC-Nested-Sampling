#ifndef CHMC_NESTED_SAMPLING_HAMILTONIAN_H
#define CHMC_NESTED_SAMPLING_HAMILTONIAN_H

#include "LeapfrogIntegrator.h"
#include "ILikelihood.h"
#include <Eigen/Dense>
#include <functional>


// Constrained Hamiltonian class. Trajectory reflects off likelihood constraint boundary.
class Hamiltonian {
public:
    Hamiltonian(ILikelihood& likelihood, double epsilon);

    const Eigen::VectorXd& GetX() const { return mIntegrator.GetX(); };
    const Eigen::VectorXd& GetP() const { return mIntegrator.GetP(); };
    const double GetLikelihood() const { return mCurrentLikelihood; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);
    void Evolve(int steps);

private:
    ILikelihood& mLikelihood;
    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mGradient;
    double mCurrentLikelihood;
    double mLikelihoodConstraint;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
