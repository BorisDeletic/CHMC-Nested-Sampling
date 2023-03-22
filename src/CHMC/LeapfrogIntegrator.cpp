#include "LeapfrogIntegrator.h"
#include <stdexcept>


LeapfrogIntegrator::LeapfrogIntegrator(const double epsilon)
    :
    mEpsilon(epsilon),
    mXUpdatedBeforeP(false)
{
}


Eigen::VectorXd
LeapfrogIntegrator::UpdateX(const Eigen::VectorXd &x, const Eigen::VectorXd &p, const Eigen::VectorXd &a, const Eigen::VectorXd& metric) {
    mHalfstepP = p + 0.5 * mEpsilon * a;

    Eigen::VectorXd newX = x + mEpsilon * metric.cwiseInverse().asDiagonal() * mHalfstepP;

    mXUpdatedBeforeP = true;

    return newX;
}

// mHalfstepP must be calculated using previous position for correctness.
Eigen::VectorXd
LeapfrogIntegrator::UpdateP(const Eigen::VectorXd &a) {
    if (!mXUpdatedBeforeP) {
        throw std::runtime_error("LEAPFROG_INTEGRATOR: Position X was not updated before Momentum P.");
    }

    Eigen::VectorXd newP = mHalfstepP + 0.5 * mEpsilon * a;

    mXUpdatedBeforeP = false;

    return newP;
}

void LeapfrogIntegrator::ChangeP(const Eigen::VectorXd& oldP, const Eigen::VectorXd& newP) {
    mHalfstepP.resize(oldP.size()); // no operation if halfstep == p
    mHalfstepP = mHalfstepP - oldP + newP; // change halfstepP to be retroactively calcuted with new p.

}

