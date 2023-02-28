#include "Gaussian.h"


const double GaussianLikelihood::Likelihood(const Eigen::VectorXd& theta)
{
    double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;

    loglikelihood -= ((theta.array() - mean) / var).pow(2).sum() / 2;
    return loglikelihood;
};


const Eigen::VectorXd GaussianLikelihood::Gradient(const Eigen::VectorXd& theta)
{
    return - ((theta.array() - mean) / var.pow(2)).matrix();
};


const Eigen::VectorXd GaussianPrior::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * 3 - 1.5;
}