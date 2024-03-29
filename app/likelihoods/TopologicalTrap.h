#ifndef CHMC_NESTED_SAMPLING_TOPOLOGICALTRAP_H
#define CHMC_NESTED_SAMPLING_TOPOLOGICALTRAP_H

#include "ILikelihood.h"
#include <Eigen/Dense>

class TopologicalTrap : public ILikelihood {
public:
    TopologicalTrap(int dim, double a = 50, double mu = 4);

    const double LogLikelihood(const Eigen::VectorXd& x) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& x) override;

    const int GetDimension() override { return mDim; };

private:
    const double a;
    const double mu;
    const double var = 0.8;
    const int mDim;
};


#endif //CHMC_NESTED_SAMPLING_TOPOLOGICALTRAP_H
