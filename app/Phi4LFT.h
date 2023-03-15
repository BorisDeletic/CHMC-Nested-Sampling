#ifndef CHMC_NESTED_SAMPLING_PHI4LFT_H
#define CHMC_NESTED_SAMPLING_PHI4LFT_H

#include "ILikelihood.h"

class Phi4Likelihood : public ILikelihood {
public:
    inline Phi4Likelihood(int n, double width) : n(n), priorWidth(width) {}

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd &cube) override;
    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;

    const int GetDimension() override { return n*n; };

private:
    const int n;
    const double priorWidth;
};


#endif //CHMC_NESTED_SAMPLING_PHI4LFT_H
