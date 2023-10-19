#ifndef CHMC_NESTED_SAMPLING_EGGBOX_H
#define CHMC_NESTED_SAMPLING_EGGBOX_H

#include "ILikelihood.h"

class Eggbox : public ILikelihood {
public:
    Eggbox(int dim);

    const double LogLikelihood(const Eigen::VectorXd &theta) override;
    const Eigen::VectorXd Gradient(const Eigen::VectorXd &theta) override;

    const int GetDimension() override { return mDim; };

private:
    const int mDim;
    double tmax = 5.0 * M_PI;
};


#endif //CHMC_NESTED_SAMPLING_EGGBOX_H
