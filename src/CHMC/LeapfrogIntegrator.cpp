#include "LeapfrogIntegrator.h"
#include <stdexcept>
#include <iostream>


LeapfrogIntegrator::LeapfrogIntegrator(IParams& params)
    :
    mParams(params),
    mXUpdatedBeforeP(false)
{
}

// Epsilon factor allows integrator to use scaled epsilon e' = m*e for one integration step.
// Used for reflections.
Eigen::VectorXd
LeapfrogIntegrator::UpdateX(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const Eigen::VectorXd &a, double epsilonFactor) {
    const Eigen::VectorXd& invMetric = mParams.GetMetric().cwiseInverse();
    mLastEpsilon = mParams.GetEpsilon() * epsilonFactor;

    mHalfstepP = p + 0.5 * mLastEpsilon * a;

    Eigen::VectorXd newX = x + mLastEpsilon * invMetric.asDiagonal() * mHalfstepP;

    mXUpdatedBeforeP = true;

    return newX;
}

// mHalfstepP must be calculated using previous position for correctness.
Eigen::VectorXd
LeapfrogIntegrator::UpdateP(const Eigen::VectorXd &a) {
    if (!mXUpdatedBeforeP) {
        throw std::runtime_error("LEAPFROG_INTEGRATOR: Position X was not updated before Momentum P.");
    }
    mXUpdatedBeforeP = false;

    Eigen::VectorXd newP = mHalfstepP + 0.5 * mLastEpsilon * a;

    return newP;
}

void LeapfrogIntegrator::ChangeP(const Eigen::VectorXd& oldP, const Eigen::VectorXd& newP) {
   // mHalfstepP.resize(oldP.size()); // no operation if halfstep == p
    mHalfstepP = mHalfstepP - oldP + newP; // change halfstepP to be retroactively calcuted with new p.
}

