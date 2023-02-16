#include "LeapfrogIntegrator.h"
#include <stdexcept>

LeapfrogIntegrator::LeapfrogIntegrator(const double epsilon, const int dimension)
    :
    mEpsilon(epsilon),
    mXUpdatedBeforeP(false)
{
    mX.resize(dimension);
    mP.resize(dimension);
    mHalfstepP.resize(dimension);
}


void LeapfrogIntegrator::UpdateX(const Eigen::VectorXd &a) {
    mHalfstepP = mP + 0.5 * mEpsilon * a;
    mX = mX + mEpsilon * mHalfstepP;

    mXUpdatedBeforeP = true;
}

// mHalfstepP must be calculated using previous position for correctness.
void LeapfrogIntegrator::UpdateP(const Eigen::VectorXd &a) {
    if (!mXUpdatedBeforeP) {
        throw std::runtime_error("LEAPFROG: Position X was not updated before Momentum P.");
    }

    mP = mHalfstepP + 0.5 * mEpsilon * a;

    mXUpdatedBeforeP = false;
}

void LeapfrogIntegrator::SetP(const Eigen::VectorXd &p) {
    mHalfstepP = mHalfstepP - mP + p; // change halfstepP to be retroactively calcuted with new p.
    mP = p;
}

