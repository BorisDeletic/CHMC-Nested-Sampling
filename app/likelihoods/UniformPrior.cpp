#include "UniformPrior.h"

UniformPrior::UniformPrior(int dim, double width, double center)
    :
    mDim(dim),
    mWidth(width),
    mCenter(center)
{
}


const Eigen::VectorXd UniformPrior::PriorTransform(const Eigen::VectorXd &cube)
{
//    return cube;
    return cube.array() * mWidth - mWidth / 2 + mCenter;
}


const Eigen::VectorXd UniformPrior::Gradient(const Eigen::VectorXd &theta) {

//    Eigen::VectorXd gradient = theta.unaryExpr([&](double x){
//        if (x < -mWidth/2) return boundaryGradient;
//        if (x > mWidth/2) return -boundaryGradient;
//        return 0.0;
//    });

    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(mDim);

//    for (int i = 0; i < theta.size(); i++) {
//        if (theta[i] < -mWidth/2 + mCenter)
//            gradient[i] = boundaryGradient;
//
//        if (theta[i] > mWidth/2 + mCenter)
//            gradient[i] = boundaryGradient;
//    }
    // make this reflect off the boundaries

    return gradient;
}



