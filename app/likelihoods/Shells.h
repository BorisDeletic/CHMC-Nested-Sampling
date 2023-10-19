#ifndef CHMC_NESTED_SAMPLING_SHELLS_H
#define CHMC_NESTED_SAMPLING_SHELLS_H

#include "ILikelihood.h"

class Shells : public ILikelihood {
public:
    Shells(int dim, double radius, double width, double center);

    const double LogLikelihood(const Eigen::VectorXd &theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd &theta) override;

    const int GetDimension() override { return mDim; };

private:
    double LogAdd(double logA, double logB);

    double LogCircle(const Eigen::VectorXd &theta, double center);
    const Eigen::VectorXd GradientCircle(const Eigen::VectorXd &theta, double center);

    const int mDim;

    const double r;
    const double w;
    const double c;
};


#endif //CHMC_NESTED_SAMPLING_SHELLS_H
