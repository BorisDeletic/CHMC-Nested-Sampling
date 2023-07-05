#ifndef CHMC_NESTED_SAMPLING_GAUSSIAN_H
#define CHMC_NESTED_SAMPLING_GAUSSIAN_H

#include "ILikelihood.h"
#include "CHMC.h"


class GaussianLikelihood : public ILikelihood {
public:
    inline GaussianLikelihood(const Eigen::VectorXd mean, const Eigen::VectorXd var)
        : mean(mean.array()), var(var.array()) {}

    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;

    const int GetDimension() override { return mean.size(); };

private:
    const Eigen::ArrayXd mean;
    const Eigen::ArrayXd var;
};



#endif //CHMC_NESTED_SAMPLING_GAUSSIAN_H
