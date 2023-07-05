#ifndef CHMC_NESTED_SAMPLING_ROSENBROCK_H
#define CHMC_NESTED_SAMPLING_ROSENBROCK_H

#include "ILikelihood.h"

class RosenbrockLikelihood : public ILikelihood {
public:
    inline RosenbrockLikelihood(const int D, const double A)
            : mDim(D), A(A) {}

    const double LogLikelihood(const Eigen::VectorXd& theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) override;

    const int GetDimension() override { return mDim; };

private:
    const int mDim;
    const double A;
};


#endif //CHMC_NESTED_SAMPLING_ROSENBROCK_H
