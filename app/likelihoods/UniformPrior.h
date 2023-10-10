#ifndef CHMC_NESTED_SAMPLING_UNIFORMPRIOR_H
#define CHMC_NESTED_SAMPLING_UNIFORMPRIOR_H

#include "IPrior.h"
#include "Phi4Likelihood.h"

class UniformPrior : public IPrior {
public:
    UniformPrior(int dim, double width);

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd &cube) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd &theta) override;

    const int GetDimension() override { return mDim; };

private:
    const int mDim;
    const double mWidth;

    const double boundaryGradient = 10000;

    Phi4Likelihood phi4;
};


#endif //CHMC_NESTED_SAMPLING_UNIFORMPRIOR_H
