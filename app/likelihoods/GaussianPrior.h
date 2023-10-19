#ifndef CHMC_NESTED_SAMPLING_GAUSSIANPRIOR_H
#define CHMC_NESTED_SAMPLING_GAUSSIANPRIOR_H


#include "IPrior.h"

class GaussianPrior : public IPrior {
public:
    GaussianPrior(int dim, double width);

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd &cube) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd &theta) override;

    const int GetDimension() override { return mDim; };

private:
    const int mDim;
    const double mWidth;

};


#endif //CHMC_NESTED_SAMPLING_GAUSSIANPRIOR_H
