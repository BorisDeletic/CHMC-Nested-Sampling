#include "types.h"
#include "ILikelihood.h"
#include <Eigen/Dense>
#include <random>

class Hamiltonian;

// Constrained HMC
class CHMC {
public:
    CHMC(ILikelihood&, double epsilon, int pathLength);
    ~CHMC();

    MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint);
    const int GetDimension() { mLikelihood.GetDimension(); };
private:
    Eigen::VectorXd SampleMomentum(int size);

    ILikelihood& mLikelihood;
    std::unique_ptr<Hamiltonian> mHamiltonian;

    std::normal_distribution<double> mNorm;
    std::random_device rd;
    std::mt19937 gen;

    const int mPathLength;
};


