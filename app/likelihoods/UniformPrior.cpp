#include "UniformPrior.h"
#include "Phi4Likelihood.h"

UniformPrior::UniformPrior(int dim, double width)
    :
    mDim(dim),
    mWidth(width),
    phi4(20,0.4,0.2)
{
}


const Eigen::VectorXd UniformPrior::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube;
   // return cube.array() * mWidth - mWidth / 2;
}


const Eigen::VectorXd UniformPrior::Gradient(const Eigen::VectorXd &theta) {

//    Eigen::VectorXd gradient = theta.unaryExpr([&](double x){
//        if (x < -mWidth/2) return boundaryGradient;
//        if (x > mWidth/2) return -boundaryGradient;
//        return 0.0;
//    });

    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(mDim);
    // make this reflect off the boundaries

    return gradient;
}



