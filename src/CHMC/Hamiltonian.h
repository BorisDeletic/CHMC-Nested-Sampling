#ifndef CHMC_NESTED_SAMPLING_HAMILTONIAN_H
#define CHMC_NESTED_SAMPLING_HAMILTONIAN_H

#include <Eigen/Dense>
#include "LeapfrogIntegrator.h"


// Constrained Hamiltonian class. Trajectory reflects off likelihood constraint boundary.
class Hamiltonian {
public:
    Hamiltonian(double epsilon);

    const Eigen::VectorXd& GetX() const { return mIntegrator.GetX(); };
    const Eigen::VectorXd& GetP() const { return mIntegrator.GetP(); };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, const double likelihoodConstraint);
    void Evolve(int steps);
private:
    void UpdateGradient(const Eigen::VectorXd& x);

    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mGradient;
    double mLikelihoodConstraint;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
