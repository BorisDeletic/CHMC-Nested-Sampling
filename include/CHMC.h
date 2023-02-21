#include "types.h"
#include "ILikelihood.h"
#include "Hamiltonian.h"
#include <Eigen/Dense>
#include <random>

// Constrained HMC
class CHMC {
public:
    CHMC(ILikelihood&, double epsilon, int pathLength);

    MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint);
private:
    Eigen::VectorXd SampleMomentum(int size);

    Hamiltonian mHamiltonian;

    const int mPathLength;
    std::normal_distribution<double> mNorm;
    std::random_device rd;
    std::mt19937 gen;
};


