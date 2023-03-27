#ifndef CHMC_NESTED_SAMPLING_CHMC_H
#define CHMC_NESTED_SAMPLING_CHMC_H

#include "ISampler.h"
#include "ILikelihood.h"
#include "Hamiltonian.h"
#include "types.h"
#include <Eigen/Dense>
#include <random>

class Hamiltonian;

// Constrained HMC
class CHMC : public ISampler {
public:
    CHMC(ILikelihood&, IParams&);

    const MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint) override;
    const Rejections GetRejections();

private:
    Eigen::VectorXd SampleP(int size);

    ILikelihood& mLikelihood;
    IParams& mParams;
    Hamiltonian mHamiltonian;

    std::normal_distribution<double> mNorm;
    std::uniform_real_distribution<double> mUniform;
    std::random_device rd;
    std::mt19937 gen;

    const int mWarmupSteps = 75;
    const double inf = 1e100;

    int mReflectRejections = 0;
    int mEnergyRejections = 0;
    int mIters = 0;
};

#endif //CHMC_NESTED_SAMPLING_CHMC_H

