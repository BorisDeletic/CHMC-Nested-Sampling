#include <Eigen/Dense>


// Leapfrog Integrator. Solves Hamiltons equations for conservative force.
// X is position vector. P is momentum vector. A is acceleration vector
class LeapfrogIntegrator
{
public:
    LeapfrogIntegrator(double epsilon);

    // Must update x first with a(x) and then p using a(x_new).
    Eigen::VectorXd UpdateX(const Eigen::VectorXd& x, const Eigen::VectorXd& p, const Eigen::VectorXd& a, const Eigen::VectorXd& metric);
    Eigen::VectorXd UpdateP(const Eigen::VectorXd& a);

    void ChangeP(const Eigen::VectorXd& oldP, const Eigen::VectorXd& newP);

    void SetEpsilon(const double epsilon) { mEpsilon = epsilon; };
    const double GetEpsilon() const { return mEpsilon; };

private:
    Eigen::VectorXd mHalfstepP;

    double mEpsilon;
    bool mXUpdatedBeforeP;
};

