#ifndef CHMC_NESTED_SAMPLING_GAUSSIAN_H
#define CHMC_NESTED_SAMPLING_GAUSSIAN_H

#include "ILikelihood.h"
#include "CHMC.h"


class GaussianLikelihood : public ILikelihood {
public:
    inline GaussianLikelihood(const Eigen::VectorXd mean, const Eigen::VectorXd var, const double width)
        : mean(mean.array()), var(var.array()), priorWidth(width) {}

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd& cube) override;
    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;
    const int GetDimension() override { return mean.size(); };

private:
    const Eigen::ArrayXd mean;
    const Eigen::ArrayXd var;
    const double priorWidth;
};



#endif //CHMC_NESTED_SAMPLING_GAUSSIAN_H
