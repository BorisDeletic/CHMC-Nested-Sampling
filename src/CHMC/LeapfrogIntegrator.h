#include <Eigen/Dense>

// Leapfrog Integrator. Solves Hamiltons equations for conservative force.
// X is position vector. P is momentum vector. A is acceleration vector
class LeapfrogIntegrator
{
public:
    LeapfrogIntegrator(const double epsilon, const int dimension);

    const Eigen::VectorXd& GetX() const { return mX; };
    const Eigen::VectorXd& GetP() const { return mP; };

    void SetX(const Eigen::VectorXd& x) { mX = x; };
    void SetP(const Eigen::VectorXd& p);
    // Must update x first with a(x) and then p using a(x_new).
    void UpdateX(const Eigen::VectorXd& a);
    void UpdateP(const Eigen::VectorXd& a);
private:
    Eigen::VectorXd mX;
    Eigen::VectorXd mP;
    Eigen::VectorXd mHalfstepP;

    double mEpsilon;
    bool mXUpdatedBeforeP;
};

