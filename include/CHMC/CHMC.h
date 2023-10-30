#ifndef CHMC_NESTED_SAMPLING_CHMC_H
#define CHMC_NESTED_SAMPLING_CHMC_H

#include "ISampler.h"
#include "IPrior.h"
#include "ILikelihood.h"
#include "Hamiltonian.h"
#include "types.h"
#include <Eigen/Dense>
#include <random>

class Hamiltonian;

// Constrained HMC
class CHMC : public ISampler {
public:
    CHMC(IPrior&, ILikelihood&, IParams&);

    const MCPoint SamplePoint(const MCPoint& old, double likelihoodConstraint) override;

private:
    Eigen::VectorXd SampleP(int size);

    IParams& mParams;
    ILikelihood& mLikelihood;
    Hamiltonian mHamiltonian;

    std::normal_distribution<double> mNorm;
    std::uniform_real_distribution<double> mUniform;
    std::random_device rd;
    std::mt19937 gen;

    int mIters = 0;
    int mPointID = 1;
};

#endif //CHMC_NESTED_SAMPLING_CHMC_H

