#ifndef CHMC_NESTED_SAMPLING_GAUSSIAN_H
#define CHMC_NESTED_SAMPLING_GAUSSIAN_H

#include "ILikelihood.h"
#include "IPrior.h"
#include "CHMC.h"


class GaussianLikelihood : public ILikelihood {
public:
    inline GaussianLikelihood(const Eigen::VectorXd mean, const Eigen::VectorXd var)
        : mean(mean.array()), var(var.array()) {}

    const double Likelihood(const Eigen::VectorXd& theta);
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta);
    const int GetDimension() { return mean.size(); };

private:
    const Eigen::ArrayXd mean;
    const Eigen::ArrayXd var;
};


class GaussianPrior : public IPrior {
public:
    inline GaussianPrior(int dims) : mDims(dims) {}

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd& cube);
    const int GetDimension() { return mDims; };

private:
    const int mDims;
};


#endif //CHMC_NESTED_SAMPLING_GAUSSIAN_H
