#include "Rosenbrock.h"
#include "Likelihood.h"

const double RosenbrockLikelihood::LogLikelihood(const Eigen::VectorXd& theta)
{
    double loglikelihood = - var.log().sum() - var.size() * std::log(2 * M_PI) / 2;

    loglikelihood -= ((theta.array() - mean) / var).pow(2).sum() / 2;
    return loglikelihood;
};


const Eigen::VectorXd RosenbrockLikelihood::Gradient(const Eigen::VectorXd& theta)
{
    return - ((theta.array() - mean) / var.pow(2)).matrix();
};


const Eigen::VectorXd RosenbrockLikelihood::PriorTransform(const Eigen::VectorXd &cube)
{
    return cube.array() * priorWidth - priorWidth / 2;
}