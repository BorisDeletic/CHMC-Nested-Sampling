#include <Eigen/Dense>


// Leapfrog Integrator. Solves Hamiltons equations for conservative force.
// X is position vector. P is momentum vector. A is acceleration vector
class LeapfrogIntegrator
{
public:
    LeapfrogIntegrator(double epsilon);

    // Must update x first with a(x) and then p using a(x_new).
    Eigen::VectorXd UpdateX(const Eigen::VectorXd& x, const Eigen::VectorXd& p, const Eigen::VectorXd& a);
    Eigen::VectorXd UpdateP(const Eigen::VectorXd& x, const Eigen::VectorXd& p, const Eigen::VectorXd& a);

    void ChangeP(const Eigen::VectorXd& oldP, const Eigen::VectorXd& newP);
private:
    Eigen::VectorXd mHalfstepP;

    const double mEpsilon;
    bool mXUpdatedBeforeP;
};

