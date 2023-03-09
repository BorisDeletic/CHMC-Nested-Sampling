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

    const Eigen::VectorXd& GetX() const { return mX; };
    const Eigen::VectorXd& GetP() const { return mP; };
    const double GetEnergy() const {return mEnergy; };
    const double GetLikelihood() const { return mLogLikelihood; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);
    void Evolve();

private:
    Eigen::VectorXd ReflectP(const Eigen::VectorXd& incidentP, const Eigen::VectorXd& normal);

    ILikelihood& mLikelihood;
    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mGradient;
    double mLogLikelihood;
    double mLikelihoodConstraint;

    Eigen::VectorXd mX;
    Eigen::VectorXd mP;
    double mEnergy;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
