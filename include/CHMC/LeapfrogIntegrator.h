#ifndef CHMC_NESTED_SAMPLING_LEAPFROGINTEGRATOR_H
#define CHMC_NESTED_SAMPLING_LEAPFROGINTEGRATOR_H

#include "IParams.h"
#include <Eigen/Dense>


// Leapfrog Integrator. Solves Hamiltons equations for conservative force.
// X is position vector. P is momentum vector. A is acceleration vector
class LeapfrogIntegrator
{
public:
    LeapfrogIntegrator(IParams&);

    // Must update x first with a(x) and then p using a(x_new).
    Eigen::VectorXd UpdateX(const Eigen::VectorXd& x, const Eigen::VectorXd& p, const Eigen::VectorXd& a, double epsilonFactor = 1);
    Eigen::VectorXd UpdateP(const Eigen::VectorXd& a);

    void ChangeP(const Eigen::VectorXd& oldP, const Eigen::VectorXd& newP);

private:
    IParams& mParams;
    Eigen::VectorXd mHalfstepP;

    double mLastEpsilon;
    bool mXUpdatedBeforeP;
};


#endif //CHMC_NESTED_SAMPLING_LEAPFROGINTEGRATOR_H
