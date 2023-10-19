#include "GaussianPrior.h"


GaussianPrior::GaussianPrior(int dim, double width)
    : mDim(dim), mWidth(width)
{
}


const Eigen::VectorXd GaussianPrior::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * mWidth - mWidth / 2;
}



const Eigen::VectorXd GaussianPrior::Gradient(const Eigen::VectorXd &theta) {
    return -theta;
}
