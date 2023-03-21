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
    const Eigen::VectorXd& GetMetric() const { return mMetric; };
    const double GetEpsilon() const { return mIntegrator.GetEpsilon(); };
    const double GetLikelihood() const { return mLogLikelihood; };
    const double GetEnergy() const;

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);
    void SetMetric(const Eigen::VectorXd metric) { mMetric = metric; };
    void SetEpsilon(const double epsilon) { mIntegrator.SetEpsilon(epsilon); };

    void Evolve();

private:
    void ReflectP(const Eigen::VectorXd& normal);
    void ReflectX();

    ILikelihood& mLikelihood;
    LeapfrogIntegrator mIntegrator;
    Eigen::VectorXd mMetric;

    Eigen::VectorXd mGradient;
    double mLogLikelihood;
    double mLikelihoodConstraint;
    const int mDimension;

    const int mEpsilonReflectionLimit = 15; // Number of times Epsilon can be halved.
    int mFailedReflections = 0;

    Eigen::VectorXd mX;
    Eigen::VectorXd mP;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
