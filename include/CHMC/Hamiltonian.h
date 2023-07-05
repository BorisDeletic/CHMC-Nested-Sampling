#ifndef CHMC_NESTED_SAMPLING_HAMILTONIAN_H
#define CHMC_NESTED_SAMPLING_HAMILTONIAN_H

#include "IPrior.h"
#include "ILikelihood.h"
#include "IParams.h"
#include "LeapfrogIntegrator.h"
#include <Eigen/Dense>

// Constrained Hamiltonian class. Trajectory reflects off likelihood constraint boundary.
class Hamiltonian {
public:
    Hamiltonian(IPrior&, ILikelihood&, IParams&);

    const Eigen::VectorXd& GetX() const { return mX; };
    const Eigen::VectorXd& GetP() const { return mP; };
    const double GetLikelihood() const { return mLogLikelihood; };
    const double GetEnergy() const;
    const bool GetRejected() const { return mRejected; };
    const int GetReflections() const { return mReflections; };
    const int GetIntegrationSteps() const { return mIters; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);

    void Evolve();

private:
    void ReflectP(const Eigen::VectorXd& normal);
    void ReflectX(const Eigen::VectorXd& normal);

    IPrior& mPrior;
    ILikelihood& mLikelihood;
    IParams& mParams;

    LeapfrogIntegrator mIntegrator;

    Eigen::VectorXd mPriorGradient;
    Eigen::VectorXd mLikelihoodGradient;

    double mLogLikelihood;
    double mLikelihoodConstraint;

    bool mRejected = false;
    int mReflections;
    int mIters;

    Eigen::VectorXd mX;
    Eigen::VectorXd mP;

    const int mEpsilonReflectionLimit = 4; // Number of times Epsilon can be halved.
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
