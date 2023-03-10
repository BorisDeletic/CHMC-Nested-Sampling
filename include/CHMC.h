#ifndef CHMC_NESTED_SAMPLING_CHMC_H
#define CHMC_NESTED_SAMPLING_CHMC_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "types.h"
#include <Eigen/Dense>
#include <random>

class Hamiltonian;

// Constrained HMC
class CHMC : public ISampler {
public:
    CHMC(ILikelihood&, double epsilon, int pathLength);
    ~CHMC();

    const Eigen::VectorXd& GetMetric();

    bool WarmupAdapt(const MCPoint& init);
    const MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint);
private:
    Eigen::VectorXd SampleP(int size);
    Eigen::VectorXd CalculateVar(const Eigen::MatrixXd& samples);

    ILikelihood& mLikelihood;
    std::unique_ptr<Hamiltonian> mHamiltonian;

    std::normal_distribution<double> mNorm;
    std::uniform_real_distribution<double> mUniform;
    std::random_device rd;
    std::mt19937 gen;

    const int mPathLength;
    const int mWarmupSteps = 75;
    const double inf = 1e100;
};

#endif //CHMC_NESTED_SAMPLING_CHMC_H

