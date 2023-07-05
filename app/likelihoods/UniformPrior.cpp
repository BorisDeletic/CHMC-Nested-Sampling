#include "UniformPrior.h"


UniformPrior::UniformPrior(int dim, double width)
    :
    mDim(dim),
    mWidth(width)
{
}


const Eigen::VectorXd UniformPrior::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * mWidth - mWidth / 2;
}

const Eigen::VectorXd UniformPrior::Gradient(const Eigen::VectorXd &theta) {

    // make this reflect off the boundaries
    return Eigen::VectorXd::Zero(mDim);
}



