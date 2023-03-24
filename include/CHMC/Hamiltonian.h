#ifndef CHMC_NESTED_SAMPLING_HAMILTONIAN_H
#define CHMC_NESTED_SAMPLING_HAMILTONIAN_H

#include "LeapfrogIntegrator.h"
#include "ILikelihood.h"
#include "IParams.h"
#include <Eigen/Dense>
#include <functional>

// Constrained Hamiltonian class. Trajectory reflects off likelihood constraint boundary.
class Hamiltonian {
public:
    Hamiltonian(ILikelihood&, IParams&);

    const Eigen::VectorXd& GetX() const { return mX; };
    const Eigen::VectorXd& GetP() const { return mP; };
    const double GetLikelihood() const { return mLogLikelihood; };
    const double GetEnergy() const;
    const bool GetRejected() const { return mRejected; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);

    void Evolve();

private:
    void ReflectP(const Eigen::VectorXd& normal);
    void ReflectX(const Eigen::VectorXd& normal);

    ILikelihood& mLikelihood;
    IParams& mParams;
    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mGradient;
    double mLogLikelihood;
    double mLikelihoodConstraint;
    const int mDimension;

    const int mEpsilonReflectionLimit = 15; // Number of times Epsilon can be halved.
    int mFailedReflections = 0;
    bool mRejected = false;

    Eigen::VectorXd mX;
    Eigen::VectorXd mP;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
