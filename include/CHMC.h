#include "types.h"
#include "ILikelihood.h"
#include <Eigen/Dense>
#include <random>

class Hamiltonian;

// Constrained HMC
class CHMC {
public:
    CHMC(ILikelihood&, double epsilon, int pathLength);

    MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint);
private:
    Eigen::VectorXd SampleMomentum(int size);

    std::unique_ptr<Hamiltonian> mHamiltonian;

    const int mPathLength;
    std::normal_distribution<double> mNorm;
    std::random_device rd;
    std::mt19937 gen;
};


