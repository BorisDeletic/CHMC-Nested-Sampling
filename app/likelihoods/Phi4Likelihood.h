#ifndef CHMC_NESTED_SAMPLING_PHI4LIKELIHOOD_H
#define CHMC_NESTED_SAMPLING_PHI4LIKELIHOOD_H

#include "ILikelihood.h"

class Phi4Likelihood : public ILikelihood {
public:
    inline Phi4Likelihood(int n, double kappa, double lambda)
    : n(n), mKappa(kappa), mLambda(lambda) {}

    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;

    const Eigen::VectorXd DerivedParams(const Eigen::VectorXd& theta) override;
    const std::vector<std::string> ParamNames() override;

    const int GetDimension() override { return n*n; };

private:
    double Potential(double field);
    double NeighbourSum(const Eigen::VectorXd& theta, int i, int j);

    const Eigen::VectorXd SpatialCorrelation(const Eigen::VectorXd& theta);
    const Eigen::VectorXd SpatialCorrelationFFT(const Eigen::VectorXd& theta);


    const int n;

    const double mKappa;
    const double mLambda;
};


#endif //CHMC_NESTED_SAMPLING_PHI4LIKELIHOOD_H
