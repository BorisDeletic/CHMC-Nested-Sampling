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
    double GetLikelihood() const { return mLogLikelihood; };
    double GetEnergy() const;
    bool GetRejected() const { return mRejected; };
    int GetReflections() const { return mReflections; };
    int GetIntegrationSteps() const { return mIters; };

//    const std::vector<double>& GetDxs() const { return dxs; };
//    const std::vector<double>& GetLikes() const { return likes; };
//    const std::vector<double>& GetProposedLikes() const { return proposed_likes; };
//    const std::vector<double>& GetPathEnergies() const { return energies; };
//    const std::vector<double>& GetMomentums() const { return momentums; };

    void SetHamiltonian(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double likelihoodConstraint);

    void Evolve();

private:
    void ReflectP(const Eigen::VectorXd& normal);
    void Reflection();

    bool OutsidePriorBounds(const Eigen::VectorXd& theta);
    const Eigen::VectorXd GetPriorReflection(const Eigen::VectorXd& theta);

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

    const int mEpsilonReflectionLimit = 10; // Number of times Epsilon can be halved.

    // FOR DIAGNOSTICS
//    std::vector<double> dxs;
//    std::vector<double> likes;
//    std::vector<double> energies;
//    std::vector<double> momentums;
//    std::vector<double> proposed_likes;

    bool mUsePosterior = true;
};


#endif //CHMC_NESTED_SAMPLING_HAMILTONIAN_H
