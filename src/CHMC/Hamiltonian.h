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
    const Eigen::MatrixXd& GetMetric() const { return mMetric; };
    const double GetEnergy();
    const double GetLikelihood() const { return mLogLikelihood; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);
    void SetMetric(const Eigen::MatrixXd& metric) { mMetric = metric; };
    void Evolve();

private:
    Eigen::VectorXd ReflectP(const Eigen::VectorXd& incidentP, const Eigen::VectorXd& normal);

    ILikelihood& mLikelihood;
    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mGradient;
    double mLogLikelihood;
    double mLikelihoodConstraint;
    const int mDimension;

    Eigen::VectorXd mX;
    Eigen::VectorXd mP;
    Eigen::MatrixXd mMetric;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
