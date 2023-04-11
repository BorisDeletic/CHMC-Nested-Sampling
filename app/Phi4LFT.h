#ifndef CHMC_NESTED_SAMPLING_PHI4LFT_H
#define CHMC_NESTED_SAMPLING_PHI4LFT_H

#include "ILikelihood.h"

class Phi4Likelihood : public ILikelihood {
public:
    inline Phi4Likelihood(int n, double kappa, double lambda, double width)
    : n(n), mKappa(kappa), mLambda(lambda), priorWidth(width) {}

    const Eigen::VectorXd PriorTransform(const Eigen::VectorXd &cube) override;
    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd DerivedParams(const Eigen::VectorXd& theta) override;

    const int GetDimension() override { return n*n; };

private:
    double Potential(double field);
    double Laplacian(const Eigen::VectorXd& theta, int i, int j);
    double NeighbourSum(const Eigen::VectorXd& theta, int i, int j);

    const int n;
    const double priorWidth;

    const double mKappa;
    const double mLambda;
};


#endif //CHMC_NESTED_SAMPLING_PHI4LFT_H
